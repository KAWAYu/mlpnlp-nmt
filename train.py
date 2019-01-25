#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
from io import open
import numpy as np
import math
import sys
import pickle
import random
import time

import chainer
import chainer.functions as chainF
from chainer.backends import cuda
from chainer import optimizers, serializers

import chainermn

from data import PrepareData
from models import EncoderDecoderAttention

comm = None


def create_communicator():
    global comm
    comm = chainermn.create_communicator('pure_nccl')
    device = comm.intra_rank
    cuda.get_device_from_id(device).use()


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-gpu', default=-1, type=int, help="GPU ID (negative value indicates CPU)")
    parser.add_argument('-mode', '-M', default='train', type=str, help="train or test mode")
    parser.add_argument('-verbose', '-V', default=1, type=int)
    parser.add_argument('-embed_size', '-D', default=512, type=int)
    parser.add_argument('-hidden_size', '-H', default=512, type=int)
    parser.add_argument('-num_layers', '-N', default=2, type=int)
    parser.add_argument('-epoch', '-E', default=13, type=int)
    parser.add_argument('-batch_size', '-B', default=128, type=int)

    parser.add_argument('-enc_vocab', '-ev', default='', type=str)
    parser.add_argument('-dec_vocab', '-dv', default='', type=str)
    parser.add_argument('-train_src', '-ts', default='', type=str)
    parser.add_argument('-train_tgt', '-tt', default='', type=str)
    parser.add_argument('-valid_src', '-vs', default='', type=str)
    parser.add_argument('-valid_tgt', '-vt', default='', type=str)

    parser.add_argument('-learning_rate', '-lrate', default=1.0, type=float)
    parser.add_argument('-learning_rate_decay_from', '-lrate_decay_from', default=9, type=int)
    parser.add_argument('-learning_rate_decay_rate', '-lrate_drate', default=0.5, type=float)

    parser.add_argument('-optimizer', '-optim', default='SGD')
    parser.add_argument('-gradient_clipping', '-grad_clip', action="store_true")
    parser.add_argument('-gradient_clipping_norm', '-grad_clip_n', default=5.0, type=float)
    parser.add_argument('-weight_decay', '-wd', action="store_true")
    parser.add_argument('-weight_decay_v', '-wdv', default=1e-6, type=float)
    parser.add_argument('-dropout_rate', '-dropr', default=0.3, type=float)
    parser.add_argument('-model', default='')
    parser.add_argument('-initializer_scale', '-init_scale', default=0.1, type=float)
    parser.add_argument('-initializer_type', '-init_type', default='uniform')
    parser.add_argument('-truncate_length', '-trunc', default=30, type=int)

    parser.add_argument('-eval_frequency', '-eval_freq', default=0, type=int)

    parser.add_argument('-setting_file', '-setf', default='')
    parser.add_argument('-output_setting_file', '-setoutf', default='')

    parser.add_argument('--use-encoder-bos-eos', dest='flag_enc_boseos', default=0, type=int)
    parser.add_argument('--merge-encoder-fwbw', dest='flag_merge_encfwbw', default=0, type=int)
    parser.add_argument('--attention-mode', dest='attn_mode', default=1, type=int)
    parser.add_argument('--use-decoder-inputfeed', dest='flag_dec_ifeed', default=1, type=int)

    parser.add_argument('--shuffle-data-mode', dest='mode_data_shuffle', default=0, type=int)
    parser.add_argument('--random-seed', dest='seed', default=2723, type=int)

    return parser.parse_args()


# 主に学習時の状況を表示するための情報を保持するクラス
class TrainProcInfo:
    def __init__(self):
        self.lossVal = 0
        self.instanceNum = 0
        self.corTot = 0
        self.incorTot = 0
        self.batchCount = 0
        self.trainsizeTot = 0
        self.procTot = 0

        self.gnorm = 0
        self.gnormLimit = 0
        self.pnorm = 0

        self.encMaxLen = 0
        self.decMaxLen = 0

    def update(self, loss_stat, mbs, bc, cor, incor, tsize, proc, encLen, decLen):
        self.instanceNum += mbs  # 文数を数える
        self.batchCount += bc  # minibatchで何回処理したか
        self.corTot += cor
        self.incorTot += incor
        self.trainsizeTot += tsize
        self.procTot += proc
        # 強制的にGPUからCPUに値を移すため floatを利用
        self.lossVal += float(loss_stat)

        self.encMaxLen = max(encLen * mbs, self.encMaxLen)
        self.decMaxLen = max(decLen * mbs, self.decMaxLen)

    # 途中経過を標示するための情報取得するルーチン
    def print_strings(self, train_mode, epoch, cMBSize, encLen, decLen, start_time, args):
        with cuda.get_device(self.lossVal):
            msg0 = 'Epoch: %3d | LL: %9.6f PPL: %10.4f' % (
                epoch, float(self.lossVal / max(1, self.procTot)),
                math.exp(min(10, float(self.lossVal / max(1, self.procTot)))))
            msg1 = '| gN: %8.4f %8.4f %8.4f' % (self.gnorm, self.gnormLimit, self.pnorm)
            dt = self.corTot + self.incorTot
            msg2 = '| acc: %6.2f %8d %8d ' % (
                float(100.0 * self.corTot / max(1, dt)), self.corTot, self.incorTot)
            msg3 = '| tot: %8d proc: %8d | num: %8d %6d %6d ' % (
                self.trainsizeTot, self.procTot, self.instanceNum, self.encMaxLen, self.decMaxLen)
            msg4 = '| MB: %4d %6d %4d %4d | Time: %10.4f' % (
                cMBSize, self.batchCount, encLen, decLen, time.time() - start_time)
            # dev.dataのときは必ず評価，学習データのときはオプションに従う
            if train_mode == 0:
                msgA = '%s %s %s %s' % (msg0, msg2, msg3, msg4)
            # elif args.doEvalAcc > 0:
            msgA = '%s %s %s %s %s' % (msg0, msg1, msg2, msg3, msg4)
            # else:
            #     msgA = '%s %s %s %s' % (msg0, msg1, msg3, msg4)
            return msgA


# optimizerの準備
def setOptimizer(args, EncDecAtt):
    global comm
    # optimizerを構築
    if args.optimizer == 'SGD':
        optimizer = chainermn.create_multi_node_optimizer(optimizers.SGD(lr=args.learning_rate), comm)
        # sys.stdout.write('# SET Learning %s: initial learning rate: %e\n' % (args.optimizer, optimizer.lr))
    elif args.optimizer == 'Adam':
        # assert 0, "Currently Adam is not supported for asynchronous update"
        optimizer = chainermn.create_multi_node_optimizer(optimizers.Adam(alpha=args.learning_rate), comm)
        # sys.stdout.write('# SET Learning %s: initial learning rate: %e\n' % (args.optimizer, optimizer.alpha))
    elif args.optimizer == 'MomentumSGD':
        optimizer = chainermn.create_multi_node_optimizer(optimizers.MomentumSGD(lr=args.learning_rate), comm)
        # sys.stdout.write('# SET Learning %s: initial learning rate: %e\n' % (args.optimizer, optimizer.lr))
    elif args.optimizer == 'AdaDelta':
        optimizer = chainermn.create_multi_node_optimizer(optimizers.AdaDelta(rho=args.learning_rate), comm)
        # sys.stdout.write('# SET Learning %s: initial learning rate: %e\n' % (args.optimizer, optimizer.rho))
    else:
        assert 0, "ERROR"

    optimizer.setup(EncDecAtt.model)  # ここでoptimizerにモデルを貼り付け
    if args.optimizer == 'Adam':
        optimizer.t = 1  # warning回避のちょっとしたhack 本来はするべきではない

    return optimizer


def decoder_processor(model, optimizer, train_mode, decSent, encInfo, args):
    global comm
    # if args.gpu >= 0:
    #     comm = chainermn.create_communicator('pure_nccl')
    #     device = comm.intra_rank
    #     cuda.get_device_from_id(device).use()
    cMBSize = encInfo.cMBSize
    aList, finalHS = model.prepareDecoder(encInfo)

    xp = cuda.get_array_module(encInfo.lstmVars[0].data)
    total_loss_val = 0
    correct = 0
    incorrect = 0
    proc = 0
    decoder_proc = len(decSent) - 1
    # print("decSent:", ' '.join(model.index2decoderWord[idx[0]] for idx in decSent if idx[0] != -1))

    ##### ここから開始 ###############################################
    # 1. decoder側の入力単語embeddingsをまとめて取得
    decEmbListCopy = model.getDecoderInputEmbeddings(decSent[:decoder_proc], args)
    decSent = xp.array(decSent)

    # 2. decoder側のRNN部分を計算
    prev_h4 = None
    prev_lstm_states = None
    trunc_loss = chainer.Variable(xp.zeros((), dtype=xp.float32))
    preds = []
    for index in range(decoder_proc):
        if index == 0:
            t_lstm_states = encInfo.lstmVars
            t_finalHS = finalHS
        else:
            t_lstm_states = prev_lstm_states
            t_finalHS = prev_h4
        # decoder LSTMを一回分計算
        hOut, lstm_states = model.processDecLSTMOneStep(
            decEmbListCopy[index], t_lstm_states, t_finalHS, args, args.dropout_rate
        )
        # lstm_statesをキャッシュ
        prev_lstm_states = lstm_states
        # attentionの計算
        finalHS = model.calcAttention(hOut, encInfo.attnList, aList, encInfo.encLen, cMBSize, args)
        # finalHSをキャッシュ
        prev_h4 = finalHS

        # 3. output(softmax)層の計算
        # 2で用意したcopyを使って最終出力層の計算をする
        oVector = model.generateWord(prev_h4, encInfo.encLen, cMBSize, args, args.dropout_rate)
        # 正解データ
        correctLabel = decSent[index + 1]
        proc += xp.count_nonzero(correctLabel + 1)
        # 必ずminibatchsizeでわる (???)
        closs = chainF.softmax_cross_entropy(oVector, correctLabel, normalize=False)
        # これで正規化なしのloss  cf. seq2seq-attn code
        total_loss_val += closs.data * cMBSize
        if train_mode > 0:  # 学習データのみ backward する
            trunc_loss += closs
            # 実際の正解数を獲得したい
        t_correct = 0
        t_incorrect = 0
        # Devのときは必ず評価，学習データのときはオプションに従って評価
        if train_mode >= 0:  # or args.doEvalAcc > 0:
            # 予測した単語のID配列 CuPy
            pred_arr = oVector.data.argmax(axis=1)
            preds.append(pred_arr)
            # 正解と予測が同じなら0になるはず => 正解したところは0なので，全体から引く
            t_correct = (correctLabel.size - xp.count_nonzero(correctLabel - pred_arr))
            # 予測不要の数から正解した数を引く # +1はbroadcast
            t_incorrect = xp.count_nonzero(correctLabel + 1) - t_correct
        correct += t_correct
        incorrect += t_incorrect
        if train_mode > 0 and (index + 1) % args.truncate_length == 0:
            model.model.cleargrads()
            trunc_loss.backward()
            trunc_loss.unchain_backward()
            optimizer.update()
        ####
    if train_mode > 0 and (index + 1) % args.truncate_length != 0:  # 学習時のみ backward する
        model.model.cleargrads()
        trunc_loss.backward()
        trunc_loss.unchain_backward()
        optimizer.update()
    return total_loss_val, (correct, incorrect, decoder_proc, proc)


# 学習用のサブルーチン
def train_model_sub(train_mode, epoch, tData, EncDecAtt, optimizer, start_time, comm, args):
    if 1:  # 並列処理のコードとインデントを揃えるため．．．
        #####################
        tInfo = TrainProcInfo()
        prnCnt = 0
        #####################
        if train_mode > 0:  # train
            dropout_rate = args.dropout_rate
        else:              # dev
            dropout_rate = 0
        #####################
        if train_mode > 0:  # train
            chainer.global_config.train = True
            chainer.global_config.enable_backprop = True
            sys.stderr.write('# TRAIN epoch {} drop rate={} | CHAINER CONFIG  [{}] \n'
                             .format(epoch, dropout_rate, chainer.global_config.__dict__))
        else:              # dev
            chainer.global_config.train = False
            chainer.global_config.enable_backprop = False
            sys.stderr.write('# DEV.  epoch {} drop rate={} | CHAINER CONFIG  [{}] \n'
                             .format(epoch, dropout_rate, chainer.global_config.__dict__))
        #####################
        # メインループ
        for encSent, decSent in tData:
            try:
                ###########################
                if train_mode > 0:  # train
                    EncDecAtt.model.cleargrads()  # パラメタ更新のためにgrad初期化
                ###########################
                encInfo = EncDecAtt.encodeSentenceFWD(train_mode, encSent, args, dropout_rate, comm)
                # loss_stat, acc_stat = EncDecAtt.trainOneMiniBatch(train_mode, decSent, encInfo, args, dropout_rate, comm)
                loss_stat, acc_stat = decoder_processor(EncDecAtt, optimizer, train_mode, decSent, encInfo, args)
                ###########################
                # mini batch のiサイズは毎回違うので取得
                cMBSize = encInfo.cMBSize
                encLen = len(encSent)
                decLen = len(decSent)
                tInfo.instanceNum += cMBSize  # 文数を数える
                tInfo.batchCount += 1  # minibatchで何回処理したか
                tInfo.corTot += acc_stat[0]
                tInfo.incorTot += acc_stat[1]
                tInfo.trainsizeTot += acc_stat[2]
                tInfo.procTot += acc_stat[3]
                # 強制的にGPUからCPUに値を移すため floatを利用
                tInfo.lossVal += float(loss_stat)
                ###########################
                # if train_mode > 0:
                #     optimizer.update()  # ここでパラメタ更新
                    ###########################
                    # tInfo.gnorm = clip_obj.norm_orig
                    # tInfo.gnormLimit = clip_obj.threshold
                    # if prnCnt == 100:
                    #     # TODO 処理が重いので実行回数を減らす ロが不要ならいらない
                    #     xp = cuda.get_array_module(encInfo.lstmVars[0].data)
                    #     tInfo.pnorm = float(xp.sqrt(chainer.optimizer_hooks.gradient_clipping._sum_sqnorm(
                    #         [p.data for p in optimizer.target.params()])))
                ####################
                del encInfo
                ###################
                tInfo.encMaxLen = max(encLen * cMBSize, tInfo.encMaxLen)
                tInfo.decMaxLen = max(decLen * cMBSize, tInfo.decMaxLen)
                ###################
                if args.verbose != 0:
                    msgA = tInfo.print_strings(train_mode, epoch, cMBSize, encLen, decLen, start_time, args)
                    if train_mode > 0 and prnCnt >= 100:
                        if args.verbose > 1:
                            pass
                            # sys.stdout.write('\r')
                        # sys.stdout.write('%s\n' % msgA)
                        prnCnt = 0
                    elif args.verbose > 2:
                        pass
                        # sys.stderr.write('\n%s' % msgA)
                    elif args.verbose > 1:
                        pass
                        # sys.stderr.write('\r%s' % msgA)
                ###################
                prnCnt += 1
            except Exception as e:
                # メモリエラーなどが発生しても処理を終了せずに
                # そのサンプルをスキップして次に進める
                flag = 0
                if args.gpu >= 0:
                    import cupy
                    if isinstance(e, cupy.cuda.runtime.CUDARuntimeError):
                        cMBSize = len(encSent[0])
                        encLen = len(encSent)
                        decLen = len(decSent)
                        # sys.stdout.write('\r# GPU Memory Error? Skip! {} | enc={} dec={} mbs={} total={} | {}\n'
                        #                 .format(tInfo.batchCount, encLen, decLen, cMBSize,
                        #                         (encLen + decLen) * cMBSize, type(e)))
                        # sys.stdout.flush()
                        flag = 1
                if flag == 0:
                    # sys.stdout.write('\r# Fatal Error? {} | {} | {}\n'.format(tInfo.batchCount, type(e), e.args))
                    import traceback
                    traceback.print_exc()
                    # sys.stdout.flush()
                    sys.exit(255)
        ###########################
        return tInfo


# 学習用の関数
def train_model(args):
    global comm
    if args.setting_file:
        # sys.stdout.write('# Loading initial data  config=[%s] model=[%s] \n' %
        #                  (args.setting_file, args.init_model_file))
        EncDecAtt = pickle.load(open(args.setting_file, 'rb'))
        data = PrepareData(EncDecAtt)
    else:
        data = PrepareData(args)
        encoderVocab = pickle.load(open(args.enc_vocab, 'rb'))
        decoderVocab = pickle.load(open(args.dec_vocab, 'rb'))
        EncDecAtt = EncoderDecoderAttention(encoderVocab, decoderVocab, args)

    if args.output_setting_file:
        fout = open(args.output_setting_file + '.setting.comm%d' % comm.rank, 'wb')
        pickle.dump(EncDecAtt, fout)
        fout.close()

    # モデルの初期化
    EncDecAtt.initModel(args)  # ここでモデルをいったん初期化
    args.embed_size = EncDecAtt.eDim  # 念の為，強制置き換え
    args.hidden_size = EncDecAtt.hDim  # 念の為，強制置き換え

    # sys.stdout.write('#####################\n')
    # sys.stdout.write('# [Params] {}'.format(args))
    # sys.stdout.write('#####################\n')

    EncDecAtt.setToGPUs(args)  # ここでモデルをGPUに貼り付ける

    optimizer = setOptimizer(args, EncDecAtt)
    if args.weight_decay:
        optimizer.add_hook(chainer.optimizer.WeightDecay(args.weight_decay_v))
    if args.gradient_clipping:
        optimizer.add_hook(chainer.optimizer.GradientClipping(args.gradient_clipping_norm))

    ########################################
    # 学習済みの初期モデルがあればをここで読み込む
    if args.setting_file and args.init_model_file:
        # sys.stderr.write('Load model from: [%s]\n' % args.init_model_file)
        serializers.load_npz(args.init_model_file, EncDecAtt.model)
    else:  # 学習済みの初期モデルがなければパラメタを全初期化する
        EncDecAtt.setInitAllParameters(optimizer, init_type=args.initializer_type, init_scale=args.initializer_scale)

    encSentLenDict = data.makeSentenceLenDict(args.train_src, EncDecAtt.encoderVocab, input_side=True)
    decSentLenDict = data.makeSentenceLenDict(args.train_tgt, EncDecAtt.decoderVocab, input_side=False)
    if args.mode_data_shuffle == 0:  # default
        trainData = data.makeBatch4Train(comm, encSentLenDict, decSentLenDict, args.batch_size, shuffle_flag=True)
    if args.valid_src and args.valid_tgt:
        encSentLenDictDevel = data.makeSentenceLenDict(args.valid_src, EncDecAtt.encoderVocab, input_side=True)
        decSentLenDictDevel = data.makeSentenceLenDict(args.valid_tgt, EncDecAtt.decoderVocab, input_side=False)
        develData = data.makeBatch4Train(comm, encSentLenDictDevel, decSentLenDictDevel, args.batch_size, shuffle_flag=False)

    prev_loss_valid = 1.0e+100
    prev_acc_valid = 0
    prev_loss_train = 1.0e+100

    # 学習ループ
    for epoch in range(args.epoch):
        ####################################
        # devの評価モード
        if args.valid_src and args.valid_tgt:
            train_mode = 0
            begin = time.time()
            sys.stdout.write('# Dev. data | total mini batch bucket size = {0}\n'.format(len(develData)))
            tInfo = train_model_sub(train_mode, epoch, develData, EncDecAtt, None, begin, comm, args)
            msgA = tInfo.print_strings(train_mode, epoch, 0, 0, 0, begin, args)
            dL = prev_loss_valid - float(tInfo.lossVal)
            sys.stdout.write('\r# Dev.Data | %s | diff: %e\n' % (msgA, dL / max(1, tInfo.instanceNum)))
            # learning rateを変更するならここ
            if args.optimizer == 'SGD':
                if epoch >= args.learning_rate_decay_from or (epoch >= args.learning_rate_decay_from and
                                                    tInfo.lossVal > prev_loss_valid and tInfo.corTot < prev_acc_valid):
                    optimizer.lr = max(args.learning_rate * 0.01, optimizer.lr * args.learning_rate_decay_rate)
                sys.stdout.write('SGD Learning Rate: %s  (initial: %s)\n' % (optimizer.lr, args.learning_rate))
            elif args.optimizer == 'Adam':
                if epoch >= args.learning_rate_decay_from or (epoch >= args.learning_rate_decay_from and
                                                    tInfo.lossVal > prev_loss_valid and tInfo.corTot < prev_acc_valid):
                    optimizer.alpha = max(args.learning_rate * 0.01, optimizer.alpha * args.learning_rate_decay_rate)
                sys.stdout.write('Adam Learning Rate: t=%s lr=%s ep=%s alpha=%s beta1=%s beta2=%s\n' % (
                   optimizer.t, optimizer.lr, optimizer.epoch, optimizer.alpha, optimizer.beta1, optimizer.beta2))
            # develのlossとaccを保存
            prev_loss_valid = tInfo.lossVal
            prev_acc_valid = tInfo.corTot
        ####################################
        # 学習モード
        # shuffleしながらmini batchを全て作成する
        # epoch==0のときは長い順（メモリ足りない場合の対策 やらなくてもよい）
        train_mode = 1
        begin = time.time()
        if args.mode_data_shuffle == 0:  # default
            # encLenの長さでまとめたものをシャッフルする
            random.shuffle(trainData)
        elif args.mode_data_shuffle == 1:  # minibatchも含めてshuffle
            trainData = data.makeBatch4Train(comm, encSentLenDict, decSentLenDict, args.batch_size, True)
        # minibatchも含めてshuffle + 最初のiterationは長さ順 (debug用途)
        elif args.mode_data_shuffle == 2:
            trainData = data.makeBatch4Train(comm, encSentLenDict, decSentLenDict, args.batch_size, (epoch != 0))
        else:
            assert 0, "ERROR"
        sys.stdout.write(
           '# Train | data shuffle | total mini batch bucket size = {0} | Time: {1:10.4f}\n'.format(
               len(trainData), time.time() - begin))
        # 学習の実体
        begin = time.time()
        tInfo = train_model_sub(train_mode, epoch, trainData, EncDecAtt, optimizer, begin, comm, args)
        msgA = tInfo.print_strings(train_mode, epoch, 0, 0, 0, begin, args)
        dL = prev_loss_train - float(tInfo.lossVal)
        sys.stdout.write('\r# Train END %s | diff: %e\n' % (msgA, dL / max(1, tInfo.instanceNum)))
        prev_loss_train = tInfo.lossVal
        ####################################
        # モデルの保存
        if args.output_setting_file:
            if epoch + 1 == args.epoch or (args.eval_frequency != 0 and (epoch + 1) % args.eval_frequency == 0):
                fout = args.output_setting_file + '.epoch%s.comm%d' % (epoch + 1, comm.rank)
                try:
                    sys.stdout.write("#output model [{}]\n".format(fout))
                    serializers.save_npz(fout, copy.deepcopy(EncDecAtt.model).to_cpu(), compression=True)
                    # chaSerial.save_hdf5(
                    #    outputFileName, copy.deepcopy(
                    #        EncDecAtt.model).to_cpu(), compression=9)
                except Exception as e:
                    pass
                    # メモリエラーなどが発生しても処理を終了せずに
                    # そのサンプルをスキップして次に進める
                    # sys.stdout.write('\r# SAVE Error? Skip! {} | {}\n'.format(fout, type(e)))
                    # sys.stdout.flush()
    ####################################
    # sys.stdout.write('Done\n')


def main():
    args = parse()
    if args.gpu >= 0:
        import cupy as xp
        cuda.check_cuda_available()
        create_communicator()
        # sys.stderr.write('w/  using GPU [%d] \n' % args.gpu)
        sys.stderr.write('w/ chainerMN')
    else:
        import numpy as xp
        args.gpu = -1
        # sys.stderr.write('w/o using GPU')

    # 乱数の初期値の設定
    # sys.stderr.write('# random seed [%d] \n' % args.seed)
    np.random.seed(args.seed)
    xp.random.seed(args.seed)
    random.seed(args.seed)

    chainer.global_config.train = True
    chainer.global_config.enable_backprop = True
    chainer.global_config.use_cudnn = "always"
    chainer.global_config.type_check = True
    # sys.stderr.write('CHAINER CONFIG  [{}] \n'.format(chainer.global_config.__dict__))
    if args.dropout_rate >= 1.0 or args.dropout_rate < 0.0:
        pass
        # sys.stderr.write('Warning: dropout rate is invalid!\nDropout rate is forcibly set 1.0')
    train_model(args)


if __name__ == '__main__':
    main()

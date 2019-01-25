#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

import chainer
import chainer.links as chainL
import chainer.functions as chainF
from chainer.backends import cuda
import chainermn

from modules import NLayerCuLSTM, NStepLSTMpp, NLayerLSTM


# EncDecの本体
class EncoderDecoderAttention:
    def __init__(self, encoderVocab, decoderVocab, setting):
        self.encoderVocab = encoderVocab  # encoderの語彙
        self.decoderVocab = decoderVocab  # decoderの語彙
        # 語彙からIDを取得するための辞書
        self.index2encoderWord = {v: k for k, v in self.encoderVocab.items()}  # 実際はなくてもいい
        self.index2decoderWord = {v: k for k, v in self.decoderVocab.items()}  # decoderで利用
        self.eDim = setting.embed_size
        self.hDim = setting.hidden_size
        self.flag_dec_ifeed = setting.flag_dec_ifeed
        self.flag_enc_boseos = setting.flag_enc_boseos
        self.attn_mode = setting.attn_mode
        self.flag_merge_encfwbw = setting.flag_merge_encfwbw

        self.encVocabSize = len(encoderVocab)
        self.decVocabSize = len(decoderVocab)
        self.n_layers = setting.num_layers

    # encoder-docoderのネットワーク
    def initModel(self, args):
        sys.stderr.write(
            ('Vocab: enc=%d dec=%d embedDim: %d, hiddenDim: %d, n_layers: %d # [Params] dec inputfeed [%d] '
             '| use Enc BOS/EOS [%d] | attn mode [%d] | merge Enc FWBW [%d]\n'
             % (self.encVocabSize, self.decVocabSize, self.eDim, self.hDim, self.n_layers, self.flag_dec_ifeed,
                self.flag_enc_boseos, self.attn_mode, self.flag_merge_encfwbw)))
        self.model = chainer.Chain(
            # encoder embedding層
            encoderEmbed=chainL.EmbedID(self.encVocabSize, self.eDim),
            # decoder embedding層
            decoderEmbed=chainL.EmbedID(self.decVocabSize, self.eDim, ignore_label=-1),
            # 出力層
            decOutputL=chainL.Linear(self.hDim, self.decVocabSize),
        )
        # logに出力する際にわかりやすくするための名前付け なくてもよい
        self.model.encoderEmbed.W.name = "encoderEmbed_W"
        self.model.decoderEmbed.W.name = "decoderEmbed_W"
        self.model.decOutputL.W.name = "decoderOutput_W"
        self.model.decOutputL.b.name = "decoderOutput_b"

        if self.flag_merge_encfwbw == 0:  # default
            self.model.add_link("encLSTM_f", NStepLSTMpp(
                self.n_layers, self.eDim, self.hDim, args.dropout_rate, name="encBiLSTMpp_fw"))
            self.model.add_link("encLSTM_b", NStepLSTMpp(
                self.n_layers, self.eDim, self.hDim, args.dropout_rate, name="encBiLSTMpp_bk"))
        elif self.flag_merge_encfwbw == 1:
            self.model.add_link("encLSTM_f", NLayerCuLSTM(self.n_layers, self.eDim, self.hDim, "encBiLSTM_fw"))
            self.model.add_link("encLSTM_b", NLayerCuLSTM(self.n_layers, self.eDim, self.hDim, "encBiLSTM_bk"))
        else:
            assert 0, "ERROR"

        # input feedの種類によって次元数が変わることに対応
        if self.flag_dec_ifeed == 0:  # inputfeedを使わない
            decLSTM_indim = self.eDim
        elif self.flag_dec_ifeed == 1:  # inputfeedを使う default
            decLSTM_indim = self.eDim + self.hDim
        # if   self.flag_dec_ifeed == 2: # inputEmbを使わない (debug用)
        #    decLSTM_indim = self.hDim
        else:
            assert 0, "ERROR"

        self.model.add_link("decLSTM", NLayerLSTM(self.n_layers, decLSTM_indim, self.hDim, "decLSTM_fw"))

        # attentionの種類によってモデル構成が違うことに対応
        if self.attn_mode > 0:  # attn_mode == 1 or 2
            self.model.add_link("attnIn_L1", chainL.Linear(self.hDim, self.hDim, nobias=True))
            self.model.add_link("attnOut_L2", chainL.Linear(self.hDim + self.hDim, self.hDim, nobias=True))
            self.model.attnIn_L1.W.name = "attnIn_W"
            self.model.attnOut_L2.W.name = "attnOut_W"
        #
        if self.attn_mode == 2:  # attention == MLP
            self.model.add_link("attnM", chainL.Linear(self.hDim, self.hDim, nobias=True))
            self.model.add_link("attnSum", chainL.Linear(self.hDim, 1, nobias=True))
            self.model.attnM.W.name = "attnM_W"
            self.model.attnSum.W.name = "attnSum_W"

    #######################################
    # ネットワークの各構成要素をGPUのメモリに配置
    def setToGPUs(self, args):
        if args.gpu >= 0:
            # sys.stderr.write('# Working on GPUs [gpu=%d]\n' % args.gpu)
            # if not args.flag_emb_cpu:  # 指定があればCPU側のメモリ上に置く
            #     self.model.encoderEmbed.to_gpu(args.gpu_enc)
            # self.model.encLSTM_f.to_gpu(args.gpu)
            self.model.encLSTM_f.to_gpu()
            # self.model.encLSTM_b.to_gpu(args.gpu)
            self.model.encLSTM_b.to_gpu()

            # if not args.flag_emb_cpu:  # 指定があればCPU側のメモリ上に置く
            #     self.model.decoderEmbed.to_gpu(args.gpu_dec)
            # self.model.decLSTM.to_gpu(args.gpu)
            self.model.decLSTM.to_gpu()
            # self.model.decOutputL.to_gpu(args.gpu)
            self.model.decOutputL.to_gpu()

            if self.attn_mode > 0:
                self.model.attnIn_L1.to_gpu()
                # self.model.attnIn_L1.to_gpu(args.gpu)
                # self.model.attnOut_L2.to_gpu(args.gpu)
                self.model.attnOut_L2.to_gpu()
            if self.attn_mode == 2:
                # self.model.attnSum.to_gpu(args.gpu)
                self.model.attnSum.to_gpu()
                self.model.attnM.to_gpu()
                # self.model.attnM.to_gpu(args.gpu)
        else:
            sys.stderr.write('# NO GPUs [gpu=%d]\n' % args.gpu)

    #######################################
    def setInitAllParameters(self, optimizer, init_type="default", init_scale=0.1):
        sys.stdout.write("############ Current Parameters BEGIN\n")
        self.printAllParameters(optimizer)
        sys.stdout.write("############ Current Parameters END\n")

        if init_type == "uniform":
            sys.stdout.write("# initializer is [uniform] [%f]\n" % init_scale)
            named_params = sorted(optimizer.target.namedparams(), key=lambda x: x[0])
            for n, p in named_params:
                with cuda.get_device(p.data):
                    t_initializer = chainer.initializers.Uniform(init_scale, p.dtype)
                    p.copydata(chainer.Parameter(t_initializer, p.data.shape))
        elif init_type == "normal":
            sys.stdout.write("# initializer is [normal] [%f]\n" % init_scale)
            named_params = sorted(optimizer.target.namedparams(), key=lambda x: x[0])
            for n, p in named_params:
                with cuda.get_device(p.data):
                    t_initializer = chainer.initializers.Normal(init_scale, p.dtype)
                    p.copydata(chainer.Parameter(t_initializer, p.data.shape))
        else:  # "default"
            sys.stdout.write("# initializer is [defalit] [%f]\n" % init_scale)
            named_params = sorted(optimizer.target.namedparams(), key=lambda x: x[0])
            for n, p in named_params:
                with cuda.get_device(p.data):
                    p.data *= init_scale
        self.printAllParameters(optimizer, init_type, init_scale)
        return 0

    def printAllParameters(self, optimizer, init_type="***", init_scale=1.0):
        total_norm = 0
        total_param = 0
        named_params = sorted(optimizer.target.namedparams(), key=lambda x: x[0])
        for n, p in named_params:
            t_norm = chainer.optimizer_hooks.gradient_clipping._sum_sqnorm(p.data)
            sys.stdout.write('### {} {} {} {} {}\n'.format(p.name, p.data.ndim, p.data.shape, p.data.size, t_norm))
            total_norm += t_norm
            total_param += p.data.size
        # with cuda.get_device(total_norm):
        #     sys.stdout.write('# param size= [{}] norm = [{}] scale=[{}, {}]\n'.format(
        #         total_param, self.model.xp.sqrt(total_norm), init_type, init_scale))

    ###############################################
    # 情報を保持するためだけのクラス 主に 細切れにbackwardするための用途
    class encInfoObject:
        def __init__(self, finalHiddenVars, finalLSTMVars, encLen, cMBSize):
            self.attnList = finalHiddenVars
            self.lstmVars = finalLSTMVars
            self.encLen = encLen
            self.cMBSize = cMBSize
    ###############################################

    # encoderのembeddingを取得する関数
    def getEncoderInputEmbeddings(self, input_idx_list, args):
        # 一文一括でembeddingを取得  この方が効率が良い？
        if args.gpu >= 0:
            input_idx_variable = chainer.Variable(input_idx_list)
            input_idx_variable.to_gpu()
            encEmbList = self.model.encoderEmbed(input_idx_variable)
            encEmbList.to_gpu()
        else:
            xp = cuda.get_array_module(self.model.encoderEmbed.W.data)
            encEmbList = self.model.encoderEmbed(chainer.Variable(xp.array(input_idx_list)))
        return encEmbList

    # decoderのembeddingを取得する関数 上のgetEncoderInputEmbeddingsとほぼ同じ
    def getDecoderInputEmbeddings(self, input_idx_list, args):
        if args.gpu >= 0:
            input_idx_variable = chainer.Variable(input_idx_list)
            input_idx_variable.to_gpu()
            decEmbList = self.model.decoderEmbed(input_idx_variable)
            decEmbList.to_gpu()
        else:
            xp = cuda.get_array_module(self.model.decoderEmbed.W.data)
            decEmbList = self.model.decoderEmbed(chainer.Variable(xp.array(input_idx_list)))
        return decEmbList

    # encoder側の入力を処理する関数
    def encodeSentenceFWD(self, train_mode, sentence, args, dropout_rate, comm):
        # if args.gpu >= 0:
        #     comm = chainermn.create_communicator('pure_nccl')
        #     device = comm.intra_rank
        #     cuda.get_device_from_id(device).use()
        encLen = len(sentence)  # 文長
        cMBSize = len(sentence[0])  # minibatch size

        # 一文一括でembeddingを取得  この方が効率が良い？
        encEmbList = self.getEncoderInputEmbeddings(sentence, args)

        flag_train = (train_mode > 0)
        lstmVars = [0] * self.n_layers * 2
        if self.flag_merge_encfwbw == 0:  # fwとbwは途中で混ぜない最後で混ぜる
            hyf, cyf, fwHout = self.model.encLSTM_f(None, None, encEmbList, flag_train, args)  # 前向き
            hyb, cyb, bkHout = self.model.encLSTM_b(None, None, encEmbList[::-1], flag_train, args)  # 後向き
            for z in range(self.n_layers):
                lstmVars[2 * z] = cyf[z] + cyb[z]
                lstmVars[2 * z + 1] = hyf[z] + hyb[z]
        elif self.flag_merge_encfwbw == 1:  # fwとbwを一層毎に混ぜる
            sp = (cMBSize, self.hDim)
            for z in range(self.n_layers):
                if z == 0:  # 一層目 embeddingを使う
                    biH = encEmbList
                else:  # 二層目以降 前層の出力を使う
                    # 加算をするためにbkHoutの逆順をもとの順序に戻す
                    biH = fwHout + bkHout[::-1]
                # z層目前向き
                hyf, cyf, fwHout = self.model.encLSTM_f(z, biH, flag_train, dropout_rate, args)
                # z層目後ろ向き
                hyb, cyb, bkHout = self.model.encLSTM_b(z, biH[::-1], flag_train, dropout_rate, args)
                # それぞれの階層の隠れ状態およびメモリセルをデコーダに
                # 渡すために保持
                lstmVars[2 * z] = chainF.reshape(cyf + cyb, sp)
                lstmVars[2 * z + 1] = chainF.reshape(hyf + hyb, sp)
        else:
            assert 0, "ERROR"

        # 最終隠れ層
        if self.flag_enc_boseos == 0:  # default
            # fwHoutを[:,]しないとエラーになる？
            biHiddenStack = fwHout[:, ] + bkHout[::-1]
        elif self.flag_enc_boseos == 1:
            bkHout2 = bkHout[::-1]  # 逆順を戻す
            biHiddenStack = fwHout[1:encLen - 1, ] + bkHout2[1:encLen - 1, ]
            # BOS, EOS分を短くする TODO おそらく長さ0のものが入るとエラー
            encLen -= 2
        else:
            assert 0, "ERROR"
        # (encの単語数, minibatchの数, 隠れ層の次元) => (minibatchの数, encの単語数, 隠れ層の次元)に変更
        biHiddenStackSW01 = chainF.swapaxes(biHiddenStack, 0, 1)
        # 各LSTMの最終状態を取得して，decoderのLSTMの初期状態を作成
        lstmVars = chainF.stack(lstmVars)
        # encoderの情報をencInfoObjectに集約して返す
        retO = self.encInfoObject(biHiddenStackSW01, lstmVars, encLen, cMBSize)
        return retO

    def prepareDecoder(self, encInfo):
        self.model.decLSTM.reset_state()
        if self.attn_mode == 0:
            aList = None
        elif self.attn_mode == 1:
            aList = encInfo.attnList
        elif self.attn_mode == 2:
            aList = self.model.attnM(chainF.reshape(encInfo.attnList, (encInfo.cMBSize * encInfo.encLen, self.hDim)))
            # TODO: 効率が悪いのでencoder側に移動したい
        else:
            assert 0, "ERROR"
        xp = cuda.get_array_module(encInfo.lstmVars[0].data)
        finalHS = chainer.Variable(
            xp.zeros(encInfo.lstmVars[0].data.shape, dtype=xp.float32))  # 最初のinput_feedは0で初期化
        return aList, finalHS

    ############################
    def trainOneMiniBatch(self, train_mode, decSent, encInfo, args, dropout_rate,comm):
        # if args.gpu >= 0:
        #     comm = chainermn.create_communicator('pure_nccl')
        #     device = comm.intra_rank
        #     cuda.get_device_from_id(device).use()
        cMBSize = encInfo.cMBSize
        aList, finalHS = self.prepareDecoder(encInfo)

        xp = cuda.get_array_module(encInfo.lstmVars[0].data)
        total_loss = chainer.Variable(xp.zeros((), dtype=xp.float32))  # 初期化
        total_loss_val = 0  # float
        correct = 0
        incorrect = 0
        proc = 0
        decoder_proc = len(decSent) - 1  # ここで処理するdecoder側の単語数

        #######################################################################
        # 1, decoder側の入力単語embeddingsをまとめて取得
        decEmbListCopy = self.getDecoderInputEmbeddings(decSent[:decoder_proc], args)
        decSent = xp.array(decSent)  # GPU上に移動
        #######################################################################
        # 2, decoder側のRNN部分を計算
        # h4_list_copy = [0] * decoder_proc
        # lstm_states_list_copy = [0] * decoder_proc
        prev_h4 = None
        prev_lstm_states = None
        trunc_loss = chainer.Variable(xp.zeros((), dtype=xp.float32))
        for index in range(decoder_proc):  # decoder_len -1
            if index == 0:
                t_lstm_states = encInfo.lstmVars
                t_finalHS = finalHS
            else:
                # t_lstm_states = lstm_states_list_copy[index - 1]
                # t_finalHS = h4_list_copy[index - 1]
                t_lstm_states = prev_lstm_states
                t_finalHS = prev_h4
            # decoder LSTMを一回ぶん計算
            hOut, lstm_states = self.processDecLSTMOneStep(
                decEmbListCopy[index], t_lstm_states, t_finalHS, args, dropout_rate)
            # lstm_statesをキャッシュ
            # lstm_states_list_copy[index] = lstm_states
            prev_lstm_states = lstm_states
            # attentionありの場合 contextベクトルを計算
            finalHS = self.calcAttention(hOut, encInfo.attnList, aList, encInfo.encLen, cMBSize, args)
            # finalHSをキャッシュ
            # h4_list_copy[index] = finalHS
            prev_h4 = finalHS
        #######################################################################
        # 3, output(softmax)層の計算
        # for index in reversed(range(decoder_proc)):
            # 2で用意した copyを使って最終出力層の計算をする
            # oVector = self.generateWord(h4_list_copy[index], encInfo.encLen, cMBSize, args, dropout_rate)
            oVector = self.generateWord(prev_h4, encInfo.encLen, cMBSize, args, dropout_rate)
            # 正解データ
            correctLabel = decSent[index + 1]  # xp
            proc += (xp.count_nonzero(correctLabel + 1))
            # 必ずminibatchsizeでわる
            closs = chainF.softmax_cross_entropy(oVector, correctLabel, normalize=False)
            # これで正規化なしのloss  cf. seq2seq-attn code
            total_loss_val += closs.data * cMBSize
            if train_mode > 0:  # 学習データのみ backward する
                # total_loss += closs
                trunc_loss += closs
            # 実際の正解数を獲得したい
            t_correct = 0
            t_incorrect = 0
            # Devのときは必ず評価，学習データのときはオプションに従って評価
            if train_mode == 0:  # or args.doEvalAcc > 0:
                # 予測した単語のID配列 CuPy
                pred_arr = oVector.data.argmax(axis=1)
                # 正解と予測が同じなら0になるはず => 正解したところは0なので，全体から引く
                t_correct = (correctLabel.size - xp.count_nonzero(correctLabel - pred_arr))
                # 予測不要の数から正解した数を引く # +1はbroadcast
                t_incorrect = xp.count_nonzero(correctLabel + 1) - t_correct
            correct += t_correct
            incorrect += t_incorrect
            if train_mode > 0 and (index + 1) % args.truncate_length == 0:
                trunc_loss.backward()
        ####
        if train_mode > 0:  # 学習時のみ backward する
            total_loss.backward()

        return total_loss_val, (correct, incorrect, decoder_proc, proc)

    # decoder LSTMの計算
    def processDecLSTMOneStep(self, decInputEmb, lstm_states_in, finalHS, args, dropout_rate):
        # 1, RNN層を隠れ層の値をセット
        # （beam searchへの対応のため毎回必ずセットする）
        self.model.decLSTM.setAllLSTMStates(lstm_states_in)
        # 2, 単語埋め込みの取得とinput feedの処理
        if self.flag_dec_ifeed == 0:  # inputfeedを使わない
            wenbed = decInputEmb
        elif self.flag_dec_ifeed == 1:  # inputfeedを使う (default)
            wenbed = chainF.concat((finalHS, decInputEmb))
        # elif self.flag_dec_ifeed == 2: # decInputEmbを使わない (debug用)
        #    wenbed = finalHS
        else:
            assert 0, "ERROR"
        # 3， N層分のRNN層を一括で計算
        h1 = self.model.decLSTM.processOneStepForward(wenbed, args, dropout_rate)
        # 4, 次の時刻の計算のためにLSTMの隠れ層を取得
        lstm_states_out = self.model.decLSTM.getAllLSTMStates()
        return h1, lstm_states_out

    # attentionの計算
    def calcAttention(self, h1, hList, aList, encLen, cMBSize, args):
        # attention使わないなら入力された最終隠れ層h1を返す
        if self.attn_mode == 0:
            return h1
        # 1, attention計算のための準備
        target1 = self.model.attnIn_L1(h1)  # まず一回変換
        # (cMBSize, self.hDim) => (cMBSize, 1, self.hDim)
        target2 = chainF.expand_dims(target1, axis=1)
        # (cMBSize, 1, self.hDim) => (cMBSize, encLen, self.hDim)
        target3 = chainF.broadcast_to(target2, (cMBSize, encLen, self.hDim))
        # target3 = chaFunc.broadcast_to(chaFunc.reshape(
        #    target1, (cMBSize, 1, self.hDim)), (cMBSize, encLen, self.hDim))
        # 2, attentionの種類に従って計算
        if self.attn_mode == 1:  # bilinear
            # bilinear系のattentionの場合は，hList1 == hList2 である
            # shape: (cMBSize, encLen)
            aval = chainF.sum(target3 * aList, axis=2)
        elif self.attn_mode == 2:  # MLP
            # attnSum に通すために変形
            t1 = chainF.reshape(target3, (cMBSize * encLen, self.hDim))
            # (cMBSize*encLen, self.hDim) => (cMBSize*encLen, 1)
            t2 = self.model.attnSum(chainF.tanh(t1 + aList))
            # shape: (cMBSize, encLen)
            aval = chainF.reshape(t2, (cMBSize, encLen))
            # aval = chaFunc.reshape(self.model.attnSum(
            #    chaFunc.tanh(t1 + aList)), (cMBSize, encLen))
        else:
            assert 0, "ERROR"
        # 3, softmaxを求める
        cAttn1 = chainF.softmax(aval)   # (cMBSize, encLen)
        # 4, attentionの重みを使ってcontext vectorを作成するところ
        # (cMBSize, encLen) => (cMBSize, 1, encLen)
        cAttn2 = chainF.expand_dims(cAttn1, axis=1)
        # (1, encLen) x (encLen, hDim) の行列演算(matmul)をcMBSize回繰り返す
        #     => (cMBSize, 1, hDim)
        cAttn3 = chainF.matmul(cAttn2, hList)
        # cAttn3 = chaFunc.batch_matmul(chaFunc.reshape(
        #    cAttn1, (cMBSize, 1, encLen)), hList)
        # axis=1の次元1になっているところを削除
        context = chainF.reshape(cAttn3, (cMBSize, self.hDim))
        # 4, attentionの重みを使ってcontext vectorを作成するところ
        # こっちのやり方でも可
        # (cMBSize, scrLen) => (cMBSize, scrLen, hDim)
        # cAttn2 = chaFunc.reshape(cAttn1, (cMBSize, encLen, 1))
        # (cMBSize, scrLen) => (cMBSize, scrLen, hDim)
        # cAttn3 = chaFunc.broadcast_to(cAttn2, (cMBSize, encLen, self.hDim))
        # 重み付き和を計算 (cMBSize, encLen, hDim)
        #     => (cMBSize, hDim)  # axis=1 がなくなる
        # context = chaFunc.sum(aList * cAttn3, axis=1)
        # 6, attention時の最終隠れ層の計算
        c1 = chainF.concat((h1, context))
        c2 = self.model.attnOut_L2(c1)
        finalH = chainF.tanh(c2)
        # finalH = chaFunc.tanh(self.model.attnOut_L2(
        #    chaFunc.concat((h1, context))))
        return finalH  # context

    # 出力層の計算
    def generateWord(self, h4, encLen, cMBSize, args, dropout_rate):
        oVector = self.model.decOutputL(chainF.dropout(h4, ratio=dropout_rate))
        return oVector

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import sys
import random
import time
from io import open
import pickle
import bottleneck as bn

import numpy as np
import chainer
from chainer.backends import cuda
from chainer import serializers
import chainer.functions as chainF

from data import PrepareData

xp = np


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-model', required=True)
    parser.add_argument('-setting_file', required=True)
    parser.add_argument('-src', required=True)
    parser.add_argument('-output', default='', type=str)
    parser.add_argument('-gpu', default=-1, type=int)

    parser.add_argument('-max_length', default=200, type=int)
    parser.add_argument('--without-unk', dest='wo_unk', default=False, action='store_true')
    parser.add_argument('--beam-size', dest='beam_size', default=1, type=int)
    parser.add_argument('--without-repeat-words', dest='wo_rep_w', default=False, action='store_true')
    parser.add_argument('--length-normalized', dest='length_normalized', default=False, action='store_true')

    return parser.parse_args()


def updateBeamThreshold__2(queue, input):
    # list内の要素はlist,タプル，かつ，0番目の要素はスコアを仮定
    if len(queue) == 0:
        queue.append(input)
    else:
        # TODO 線形探索なのは面倒なので 効率を上げるためには要修正
        for i in range(len(queue)):
            if queue[i][0] <= input[0]:
                continue
            tmp = queue[i]
            queue[i] = input
            input = tmp
    return queue


def decodeByBeamFast(EncDecAtt, encSent, cMBSize, max_length, beam_size, args):
    train_mode = 0  # 評価なので
    encInfo = EncDecAtt.encodeSentenceFWD(train_mode, encSent, args, 0.0)
    if args.gpu >= 0:
        cuda.get_device_from_id(args.gpu)
    encLen = encInfo.encLen
    aList, finalHS = EncDecAtt.prepareDecoder(encInfo)

    idx_bos = EncDecAtt.decoderVocab['<s>']
    idx_eos = EncDecAtt.decoderVocab['</s>']
    idx_unk = EncDecAtt.decoderVocab['<unk>']

    if args.wo_rep_w:
        WFilter = xp.zeros((1, EncDecAtt.decVocabSize), dtype=xp.float32)
    else:
        WFilter = None
    beam = [(0, [idx_bos], idx_bos, encInfo.lstmVars, finalHS, WFilter)]
    dummy_b = (1.0e+100, [idx_bos], idx_bos, None, None, WFilter)

    for i in range(max_length + 1):  # for </s>
        newBeam = [dummy_b] * beam_size

        cMBSize = len(beam)
        #######################################################################
        # beamで分割されているものを一括処理するために miniBatchとみなして処理
        # 準備としてbeamの情報を結合
        # beam内の候補をminibatchとして扱うために，axis=0 を 1から
        # cMBSizeに拡張するためにbroadcast
        biH0 = chainF.broadcast_to(encInfo.attnList, (cMBSize, encLen, EncDecAtt.hDim))
        if EncDecAtt.attn_mode == 1:
            aList_a = biH0
        elif EncDecAtt.attn_mode == 2:
            t = chainF.broadcast_to(
                chainF.reshape(aList, (1, encLen, EncDecAtt.hDim)), (cMBSize, encLen, EncDecAtt.hDim))
            aList_a = chainF.reshape(t, (cMBSize * encLen, EncDecAtt.hDim))
            # TODO: 効率が悪いのでencoder側に移動したい
        else:
            assert 0, "ERROR"

        zipbeam = list(zip(*beam))
        # axis=1 (defaultなので不要) ==> hstack
        lstm_states_a = chainF.concat(zipbeam[3])
        # concat(a, axis=0) == vstack(a)
        finalHS_a = chainF.concat(zipbeam[4], axis=0)
        # decoder側の単語を取得
        # 一つ前の予測結果から単語を取得
        wordIndex = np.array(zipbeam[2], dtype=np.int32)
        inputEmbList = EncDecAtt.getDecoderInputEmbeddings(wordIndex, args)
        #######################################################################
        hOut, lstm_states_a = EncDecAtt.processDecLSTMOneStep(inputEmbList, lstm_states_a, finalHS_a, args, 0.0)
        # attentionありの場合 contextベクトルを計算
        next_h4_a = EncDecAtt.calcAttention(hOut, biH0, aList_a, encLen, cMBSize, args)
        oVector_a = EncDecAtt.generateWord(next_h4_a, encLen, cMBSize, args, 0.0)
        #####
        nextWordProb_a = -chainF.log_softmax(oVector_a.data).data
        if args.wo_rep_w:
            WFilter_a = xp.concat(zipbeam[4], axis=0)
            nextWordProb_a += WFilter_a
        # 絶対に出てほしくない出力を強制的に選択できないようにするために
        # 大きな値をセットする
        nextWordProb_a[:, idx_bos] = 1.0e+100  # BOS
        if args.wo_unk:  # UNKは出さない設定の場合
            nextWordProb_a[:, idx_unk] = 1.0e+100

        #######################################################################
        # beam_size個だけ使う，使いたくない要素は上の値変更処理で事前に省く
        if args.gpu >= 0:
            nextWordProb_a = nextWordProb_a.get()  # sort のためにCPU側に移動
        sortedIndex_a = bn.argpartition(nextWordProb_a, beam_size)[:, :beam_size]
        # 遅くてもbottleneckを使いたくなければ下を使う？
        # sortedIndex_a = np.argsort(nextWordProb_a)[:, :beam_size]
        #######################################################################

        for z, b in enumerate(beam):
            # まず，EOSまで既に到達している場合はなにもしなくてよい
            # (beamはソートされていることが条件)
            if b[2] == idx_eos:
                newBeam = updateBeamThreshold__2(newBeam, b)
                continue
            ##
            flag_force_eval = False
            if i == max_length:  # mode==0,1,2: free,word,char
                flag_force_eval = True

            if not flag_force_eval and b[0] > newBeam[-1][0]:
                continue
            # 3
            # 次のbeamを作るために準備
            lstm_states = lstm_states_a[:, z:z + 1, ]
            next_h4 = next_h4_a[z:z + 1, ]
            nextWordProb = nextWordProb_a[z]
            ###################################
            # 長さ制約的にEOSを選ばなくてはいけないという場合
            if flag_force_eval:
                wordIndex = idx_eos
                newProb = nextWordProb[wordIndex] + b[0]
                if args.wo_rep_w:
                    tWFilter = b[5].copy()
                    tWFilter[:, wordIndex] += 1.0e+100
                else:
                    tWFilter = b[5]
                nb = (newProb, b[1][:] + [wordIndex], wordIndex, lstm_states, next_h4, tWFilter)
                newBeam = updateBeamThreshold__2(newBeam, nb)
                continue
            # 正解が与えられている際にはこちらを使う
            # if decoderSent is not None:
            #   wordIndex = decoderSent[i]
            #   newProb =  nextWordProb[wordIndex] + b[0]
            #   if args.wo_rep_w:
            #           tWFilter = b[5].copy()
            #           tWFilter[:,wordIndex] += 1.0e+100
            #   else:
            #                tWFilter = b[5]
            #   nb = (newProb, b[1][:]+[wordIndex], wordIndex,
            #         lstm_states, next_h4, tWFilter)
            #   newBeam = updateBeamThreshold__2(newBeam, nb)
            #   continue
            # 3
            # ここまでたどり着いたら最大beam個評価する
            # 基本的に sortedIndex_a[z] は len(beam) 個しかない
            for wordIndex in sortedIndex_a[z]:
                newProb = nextWordProb[wordIndex] + b[0]
                if newProb > newBeam[-1][0]:
                    continue
                    # break
                # ここまでたどり着いたら入れる
                if args.wo_rep_w:
                    tWFilter = b[5].copy()
                    tWFilter[:, wordIndex] += 1.0e+100
                else:
                    tWFilter = b[5]
                nb = (newProb, b[1][:] + [wordIndex], wordIndex, lstm_states, next_h4, tWFilter)
                newBeam = updateBeamThreshold__2(newBeam, nb)
                #####
        ################
        # 一時刻分の処理が終わったら，入れ替える
        beam = newBeam
        if all([True if b[2] == idx_eos else False for b in beam]):
            break
        # 次の入力へ
    beam = [(b[0], b[1], b[3], b[4], [EncDecAtt.index2decoderWord[z] if z != 0 else "$UNK$"
                                      for z in b[1]]) for b in beam]

    return beam


def rerankingByLengthNormalizedLoss(beam, wposi):
    beam.sort(key=lambda b: b[0] / (len(b[wposi]) - 1))
    return beam


# テスト部本体
def ttest_model(args):
    EncDecAtt = pickle.load(open(args.setting_file, 'rb'))
    EncDecAtt.initModel(args)
    if args.setting_file and args.model:  # モデルをここで読み込む
        sys.stderr.write('Load model from: [%s]\n' % args.model)
        serializers.load_npz(args.model, EncDecAtt.model)
    else:
        assert 0, "ERROR"
    prepD = PrepareData(EncDecAtt)

    EncDecAtt.setToGPUs(args)
    sys.stderr.write('Finished loading model\n')

    sys.stderr.write('max_length is [%d]\n' % args.max_length)
    sys.stderr.write('w/o generating unk token [%r]\n' % args.wo_unk)
    sys.stderr.write('w/o generating the same words in twice [%r]\n' % args.wo_rep_w)
    sys.stderr.write('beam size is [%d]\n' % args.beam_size)
    sys.stderr.write('output is [%s]\n' % args.output if args.output else sys.stdout)

    ####################################
    decMaxLen = args.max_length

    begin = time.time()
    counter = 0

    fout = open(args.output, 'w') if args.output else sys.stdout

    # TODO: codecsでないとエラーが出る環境がある？ 要調査 不要ならioにしたい
    with open(args.src, encoding='utf-8') as f:
        # with codecs.open(args.encDataFile, encoding='utf-8') as f:
        for sentence in f:
            sentence = sentence.strip()  # stripを忘れずに．．．
            # ここでは，入力された順番で一文ずつ処理する方式のみをサポート
            sourceSentence = prepD.sentence2index(sentence, EncDecAtt.encoderVocab, input_side=True)
            sourceSentence = np.transpose(
                np.reshape(np.array(sourceSentence, dtype=np.int32), (1, len(sourceSentence))))
            # 1文ずつ処理するので，test時は基本必ずminibatch=1になる
            cMBSize = len(sourceSentence[0])
            outputBeam = decodeByBeamFast(EncDecAtt, sourceSentence, cMBSize, decMaxLen, args.beam_size, args)
            wposi = 4
            outloop = 1
            # if args.outputAllBeam > 0:
            #    outloop = args.beam_size

            # 長さに基づく正規化 このオプションを使うことを推奨
            if args.length_normalized:
                outputBeam = rerankingByLengthNormalizedLoss(outputBeam, wposi)

            for i in range(outloop):
                outputList = outputBeam[i][wposi]
                # score = outputBeam[i][0]
                if outputList[-1] != '</s>':
                    outputList.append('</s>')
                # if args.outputAllBeam > 0:
                # sys.stdout.write("# {} {} {}\n".format(i, score,
                # len(outputList)))

                print(' '.join(outputList[1:len(outputList) - 1]), file=fout)
                # sys.stdout.write('{}\n'.format(' '.join(outputList[1:len(outputList) - 1])))
                # charlenList = sum([ len(z)+1 for z in
                # 文末の空白はカウントしないので-1
                # outputList[1:len(outputList) - 1] ])-1
            counter += 1
            sys.stderr.write('\rSent.Num: %5d %s  | words=%d | Time: %10.4f ' %
                             (counter, outputList, len(outputList), time.time() - begin))
    fout.close()
    sys.stderr.write('\rDONE: %5d | Time: %10.4f\n' % (counter, time.time() - begin))


def main():
    global xp
    args = parse()

    if args.gpu >= 0:
        import cupy
        xp = cupy
        cuda.check_cuda_available()
        cuda.get_device_from_id(args.gpu).use()
        sys.stderr.write('w/  using GPU [%d] \n' % args.gpu)
    else:
        args.gpu = -1
        sys.stderr.write('w/o using GPU\n')

    chainer.global_config.train = False
    chainer.global_config.enable_backprop = False
    chainer.global_config.use_cudnn = "always"
    chainer.global_config.type_check = True
    args.dropout_rate = .0
    ttest_model(args)


if __name__ == '__main__':
    main()

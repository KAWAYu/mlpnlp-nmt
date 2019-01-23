#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from io import open
import collections
import random
import numpy as np


########################################################
# データを読み込んだりするための関数をまとめたもの
class PrepareData:
    def __init__(self, setting):
        self.flag_enc_boseos = setting.flag_enc_boseos

    ################################################
    def readVocab(self, vocabFile):  # 学習時にのみ呼ばれる予定
        d = {}
        d.setdefault('<unk>', len(d))  # 0番目 固定
        sys.stdout.write('# Vocab: add <unk> | id={}\n'.format(d['<unk>']))
        d.setdefault('<s>', len(d))   # 1番目 固定
        sys.stdout.write('# Vocab: add <s>   | id={}\n'.format(d['<s>']))
        d.setdefault('</s>', len(d))  # 2番目 固定
        sys.stdout.write('# Vocab: add </s>  | id={}\n'.format(d['</s>']))

        # TODO: codecsでないとエラーが出る環境がある？ 要調査 不要ならioにしたい
        with open(vocabFile, encoding='utf-8') as f:
            # with codecs.open(vocabFile, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                word, freq = line.split('\t')  # 基本的にtab区切りを想定
                # word, freq = line.split(' ')  # スペース区切りはこちらに変更
                if word == "<unk>":
                    continue
                elif word == "<s>":
                    continue
                elif word == "</s>":
                    continue
                d.setdefault(word, len(d))
        return d

    def sentence2index(self, sentence, word2indexDict, input_side=False):
        indexList = [word2indexDict[word] if word in word2indexDict else word2indexDict['<unk>']
                     for word in sentence.split(' ')]
        # encoder側でかつ，<s>と</s>を使わない設定の場合
        if input_side and self.flag_enc_boseos == 0:
            return indexList
        else:  # 通常はこちら
            return [word2indexDict['<s>']] + indexList + [word2indexDict['</s>']]

    def makeSentenceLenDict(self, fileName, word2indexDict, input_side=False):
        if input_side:
            d = collections.defaultdict(list)
        else:
            d = {}
        sentenceNum = 0
        sampleNum = 0
        maxLen = 0
        # ここで全てのデータを読み込む
        # TODO: codecsでないとエラーが出る環境がある？ 要調査 不要ならioにしたい
        with open(fileName, encoding='utf-8') as f:
            # with codecs.open(fileName, encoding='utf-8') as f:
            for sntNum, snt in enumerate(f):  # ここで全てのデータを読み込む
                snt = snt.strip()
                indexList = self.sentence2index(snt, word2indexDict, input_side=input_side)
                sampleNum += len(indexList)
                if input_side:
                    # input側 ここで長さ毎でまとめたリストを作成する
                    # 値は文番号と文そのもののペア
                    d[len(indexList)].append((sntNum, indexList))
                else:
                    d[sntNum] = indexList  # decoder側 文の番号をキーとしたハッシュ
                sentenceNum += 1
                maxLen = max(maxLen, len(indexList))
        sys.stdout.write('# data sent: %10d  sample: %10d maxlen: %10d\n' % (sentenceNum, sampleNum, maxLen))
        return d

    def makeBatch4Train(self, encSentLenDict, decSentLenDict, batch_size=1, shuffle_flag=True):
        encSentDividedBatch = []
        for length, encSentList in encSentLenDict.items():
            random.shuffle(encSentList)  # ここで同じencLenのデータをshuffle
            iter2 = range(0, len(encSentList), batch_size)
            encSentDividedBatch.extend([encSentList[_:_ + batch_size] for _ in iter2])
        if shuffle_flag is True:
            # encLenの長さでまとめたものをシャッフルする
            random.shuffle(encSentDividedBatch)
        else:
            sys.stderr.write('# NO shuffle: descending order based on encoder sentence length\n')

        encSentBatch = []
        decSentBatch = []
        # shuffleなしの場合にencoderの長い方から順番に生成
        for batch in encSentDividedBatch[::-1]:
            encSentBatch.append(np.array([encSent for sntNum, encSent in batch], dtype=np.int32).T)
            maxDecoderLength = max([len(decSentLenDict[sntNum]) for sntNum, encSent in batch])
            decSentBatch.append(
                np.array([decSentLenDict[sntNum] + [-1] * (maxDecoderLength - len(decSentLenDict[sntNum]))
                          for sntNum, encSent in batch], dtype=np.int32).T)
        ######
        return list(zip(encSentBatch, decSentBatch))

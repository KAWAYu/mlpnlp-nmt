#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import chainer
import chainer.links as chainL
import chainer.functions as chainF
from chainer.backends import cuda


# LSTMの層の数を変数で決定したいので，層数が可変なことをこのクラスで吸収する
# ここでは主にdecoder用のLSTMを構築するために利用
class NLayerLSTM(chainer.ChainList):
    def __init__(self, n_layers=2, eDim=512, hDim=512, name=""):
        layers = [0] * n_layers  # 層分の領域を確保
        for z in range(n_layers):
            if z == 0:  # 第一層の次元数は eDim
                tDim = eDim
            else:  # 第二層以上は前の層の出力次元が入力次元となるのでhDim
                tDim = hDim
            layers[z] = chainL.LSTM(tDim, hDim)
            # logに出力する際にわかりやすくするための名前付け
            layers[z].lateral.W.name = name + "_L%d_la_W" % (z + 1)
            layers[z].upward.W.name = name + "_L%d_up_W" % (z + 1)
            layers[z].upward.b.name = name + "_L%d_up_b" % (z + 1)

        super(NLayerLSTM, self).__init__(*layers)

    # 全ての層に対して一回だけLSTMを回す
    def processOneStepForward(self, input_states, args, dropout_rate):
        hout = None
        for c, layer in enumerate(self):
            if c > 0:  # 一層目(embedding)の入力に対してはdropoutしない
                hin = chainF.dropout(hout, ratio=dropout_rate)
            else:  # 二層目以降の入力はdropoutする
                hin = input_states
            hout = layer(hin)
        return hout

    # 全ての層を一括で初期化
    def reset_state(self):
        for layer in self:
            layer.reset_state()

    # 主に encoder と decoder 間の情報の受渡しや，beam searchの際に
    # 連続でLSTMを回せない時に一旦情報を保持するための関数
    def getAllLSTMStates(self):
        lstm_state_list_out = [0] * len(self) * 2
        for z in range(len(self)):
            lstm_state_list_out[2 * z] = self[z].c
            lstm_state_list_out[2 * z + 1] = self[z].h
        # 扱いやすくするために，stackを使って一つの Chainer Variableにして返す
        return chainF.stack(lstm_state_list_out)

    # 用途としては，上のgetAllLSTMStatesで保存したものをセットし直すための関数
    def setAllLSTMStates(self, lstm_state_list_in):
        for z in range(len(self)):
            self[z].c = lstm_state_list_in[2 * z]
            self[z].h = lstm_state_list_in[2 * z + 1]


# 組み込みのNStepLSTMを必要な形に修正したもの （cuDNNを使って高速化するため）
class NStepLSTMpp(chainer.ChainList):
    def __init__(self, n_layers, in_size, out_size, dropout_rate, name="", use_cudnn=True):
        weights = []
        direction = 1  # ここでは，からなず一方向ずつ構築するので1にする
        t_name = name
        if name is not "":
            t_name = '%s_' % name

        for i in range(n_layers):
            for di in range(direction):
                weight = chainer.Link()
                for j in range(8):
                    if i == 0 and j < 4:
                        w_in = in_size
                    elif i > 0 and j < 4:
                        w_in = out_size * direction
                    else:
                        w_in = out_size
                    weight.add_param('%sw%d' % (t_name, j), (out_size, w_in))
                    weight.add_param('%sb%d' % (t_name, j), (out_size,))
                    getattr(weight, '%sw%d' %
                            (t_name, j)).data[...] = np.random.normal(0, np.sqrt(1. / w_in), (out_size, w_in))
                    getattr(weight, '%sb%d' % (t_name, j)).data[...] = 0
                weights.append(weight)

        super(NStepLSTMpp, self).__init__(*weights)

        self.n_layers = n_layers
        self.dropout_rate = dropout_rate
        self.use_cudnn = use_cudnn
        self.out_size = out_size
        self.direction = direction
        self.ws = [[getattr(w, '%sw0' % t_name),
                    getattr(w, '%sw1' % t_name),
                    getattr(w, '%sw2' % t_name),
                    getattr(w, '%sw3' % t_name),
                    getattr(w, '%sw4' % t_name),
                    getattr(w, '%sw5' % t_name),
                    getattr(w, '%sw6' % t_name),
                    getattr(w, '%sw7' % t_name)] for w in self]
        self.bs = [[getattr(w, '%sb0' % t_name),
                    getattr(w, '%sb1' % t_name),
                    getattr(w, '%sb2' % t_name),
                    getattr(w, '%sb3' % t_name),
                    getattr(w, '%sb4' % t_name),
                    getattr(w, '%sb5' % t_name),
                    getattr(w, '%sb6' % t_name),
                    getattr(w, '%sb7' % t_name)] for w in self]

    def init_hx(self, xs):
        hx_shape = self.n_layers * self.direction
        with cuda.get_device_from_id(self._device_id):
            hx = chainer.Variable(self.xp.zeros((hx_shape, xs.data.shape[1], self.out_size), dtype=xs.dtype))
        return hx

    def __call__(self, hx, cx, xs, flag_train, args):
        if hx is None:
            hx = self.init_hx(xs)
        if cx is None:
            cx = self.init_hx(xs)

        # hx, cx は (layer数, minibatch数，出力次元数)のtensor
        # xsは (系列長, minibatch数，出力次元数)のtensor
        # Note: chainF.n_step_lstm() は最初の入力層にはdropoutしない仕様
        hy, cy, ys = chainF.n_step_lstm(self.n_layers, self.dropout_rate, hx, cx, self.ws, self.bs, xs)
        # hy, cy は (layer数, minibatch数，出力次元数) で出てくる
        # ysは最終隠れ層だけなので，系列長のタプルで
        # 各要素が (minibatch数，出力次元数)
        # 扱いやすくするためにstackを使ってタプルを一つのchainer.Variableに変換
        # (系列長, minibatch数，出力次元数)のtensor
        hlist = chainF.stack(ys)
        return hy, cy, hlist


# LSTMの層の数を変数で決定したいので，層数が可変なことをこのクラスで吸収する
class NLayerCuLSTM(chainer.ChainList):
    def __init__(self, n_layers, eDim, hDim, name=""):
        layers = [0] * n_layers
        for z in range(n_layers):
            if name is not "":  # 名前を付ける
                t_name = '%s_L%d' % (name, z + 1)
            # 毎回一層分のNStepLSTMを作成
            if z == 0:
                tDim = eDim
            else:
                tDim = hDim
            # 手動で外でdropoutするのでここではrateを0に固定する
            layers[z] = NStepLSTMpp(1, tDim, hDim, dropout_rate=0.0, name=t_name)

        super(NLayerCuLSTM, self).__init__(*layers)

    # layre_numで指定された層をinput_state_listの長さ分回す
    def __call__(self, layer_num, input_state_list, flag_train, dropout_rate, args):
        # Note: chainF.n_step_lstm() は最初の入力にはdropoutしない仕様なので，
        # 一層毎に手動で作った場合は手動でdropoutが必要
        if layer_num > 0:
            hin = chainF.dropout(input_state_list, ratio=dropout_rate)
        else:
            hin = input_state_list
        # layer_num層目の処理を一括で行う
        hy, cy, hout = self[layer_num](None, None, hin, flag_train, args)
        return hy, cy, hout

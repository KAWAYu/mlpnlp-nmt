#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from io import open
from collections import Counter
import pickle
import sys


def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-train_src', required=True, help="Source side file of training data")
    parser.add_argument('-train_tgt', required=True, help="Target side file of training data")
    parser.add_argument('-valid_src', required=True, help="Source side file of validation data")
    parser.add_argument('-valid_tgt', required=True, help="target side file of validation data")
    parser.add_argument('-save_data', required=True, help="prefix of data file")
    parser.add_argument('-threshold', default=3, type=int, help="minimum threshold of the frequency of word")
    parser.add_argument('-src_threshold', '-sthre', type=int, default=-1)
    parser.add_argument('-tgt_threshold', '-tthre', type=int, default=-1)
    parser.add_argument('-vocab_size', default=50000, type=int, help="the size of vocabulary")
    parser.add_argument('-src_vocab_size', '-svb', default=-1, type=int, help="the size of source side vocabulary")
    parser.add_argument('-tgt_vocab_size', '-tvb', default=-1, type=int, help="the size of target side vocabulary")

    return parser.parse_args()


def preprocess_args(args):
    if args.src_threshold == -1:
        args.src_threshold = args.threshold
    if args.tgt_threshold == -1:
        args.tgt_threshold = args.threshold
    if args.src_vocab_size == -1:
        args.src_vocab_size = args.vocab_size
    if args.tgt_vocab_size == -1:
        args.tgt_vocab_size = args.vocab_size
    return args


def simple_tokenizer(line):
    return line.split(' ')


def word_counter(filepath, vocab_size, threshold, tokenizer=simple_tokenizer):
    c = Counter()
    with open(filepath) as fin:
        for line in fin:
            for token in tokenizer(line.strip()):
                c[token] += 1

    vocab = {'<unk>': 0, '<s>': 1, '</s>': 2, '<pad>': 3}
    init_size = len(vocab)
    for k, v in c.most_common():
        if v <= threshold or len(vocab) >= vocab_size + init_size:
            break
        vocab[k] = len(vocab)

    return vocab


def main():
    args = parse()
    args = preprocess_args(args)
    src_vocab = word_counter(args.train_src, args.src_vocab_size, args.src_threshold)
    sys.stderr.write('Source vocabulary size: %d\n' % len(src_vocab))
    tgt_vocab = word_counter(args.train_tgt, args.tgt_vocab_size, args.tgt_threshold)
    sys.stderr.write('Target vocabulary size: %d\n' % len(tgt_vocab))
    pickle.dump(src_vocab, open(args.save_data + '.src.vocab', 'wb'))
    pickle.dump(tgt_vocab, open(args.save_data + '.tgt.vocab', 'wb'))


if __name__ == '__main__':
    main()

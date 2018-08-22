import sys
import time
import numpy as np
import random
import os

import paddle
import paddle.fluid as fluid


def to_lodtensor(data, place):
    """
    convert to LODtensor
    """
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])
    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res


def load_vocab(filename):
    """
    load imdb vocabulary
    """
    vocab = {}
    with open(filename) as f:
        wid = 0
        for line in f:
            vocab[line.strip()] = wid
            wid += 1
    vocab["<unk>"] = len(vocab)
    return vocab


def data2tensor(data, place):
    """
    data2tensor
    """
    input_seq = to_lodtensor(map(lambda x: x[0], data), place)
    y_data = np.array(map(lambda x: x[1], data)).astype("int64")
    y_data = y_data.reshape([-1, 1])
    return {"words": input_seq, "label": y_data}


def data_reader(file_path, word_dict, is_shuffle=True):
    """
    Convert word sequence into slot
    """
    unk_id = len(word_dict)
    all_data = []
    with open(file_path, "r") as fin:
        for line in fin:
            cols = line.strip().split("\t")
            label = int(cols[0])
            wids = [word_dict[x] if x in word_dict else unk_id
                    for x in cols[1].split(" ")]
            all_data.append((wids, label))
    if is_shuffle:
        random.shuffle(all_data)

    def reader():
        for doc, label in all_data:
            yield doc, label
    return reader


def prepare_data(data_type="imdb",
                 self_dict=False,
                 batch_size=128,
                 buf_size=50000):
    """
    prepare data
    """
    script_path = os.path.dirname(__file__)
    if self_dict:
        word_dict = load_vocab(data_type + ".vocab")
    else:
        if data_type == "imdb":
            word_dict = paddle.dataset.imdb.word_dict()
        elif data_type == "data":
            word_dict = load_vocab(script_path + "/data/train.vocab")
        else:
            raise RuntimeError("No such dataset")

    if data_type == "imdb":
        train_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.imdb.train(word_dict), buf_size=buf_size),
            batch_size=batch_size)
        test_reader = paddle.batch(
            paddle.reader.shuffle(
                paddle.dataset.imdb.test(word_dict), buf_size=buf_size),
            batch_size=batch_size)
    elif data_type == "data":
        train_reader = paddle.batch(
            data_reader(script_path + "/data/train_data/corpus.train", word_dict, True),
            batch_size=batch_size)
        test_reader = paddle.batch(
            # test data are corrupted (!?)
            #  data_reader(script_path + "/data/test_data/corpus.test", word_dict, False),
            data_reader(script_path + "/data/train_data/corpus.train", word_dict, True),
            batch_size=batch_size)
    else:
        raise RuntimeError("No such dataset")

    return word_dict, train_reader, test_reader

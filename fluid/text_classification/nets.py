import sys
import time
import numpy as np

import paddle.fluid as fluid
import paddle.v2 as paddle


def bow_net(data,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2,
            use_mkldnn=False):
    """
    bow net
    """
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])
    bow = fluid.layers.sequence_pool(input=emb, pool_type='sum')
    bow_tanh = fluid.layers.tanh(bow, use_mkldnn=use_mkldnn)
    fc_1 = fluid.layers.fc(input=bow_tanh, size=hid_dim, act="tanh", use_mkldnn=use_mkldnn)
    fc_2 = fluid.layers.fc(input=fc_1, size=hid_dim2, act="tanh", use_mkldnn=use_mkldnn)
    prediction = fluid.layers.fc(input=[fc_2], size=class_dim, act="softmax", use_mkldnn=use_mkldnn)
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, acc, prediction


def cnn_net(data,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2,
            win_size=3,
            use_mkldnn=False):
    """
    conv net
    """
    emb = fluid.layers.embedding(input=data, size=[dict_dim, emb_dim])

    conv_3 = fluid.nets.sequence_conv_pool(
        input=emb,
        num_filters=hid_dim,
        filter_size=win_size,
        act="tanh",
        pool_type="max")

    fc_1 = fluid.layers.fc(input=[conv_3], size=hid_dim2, use_mkldnn=use_mkldnn)

    prediction = fluid.layers.fc(input=[fc_1], size=class_dim, act="softmax", use_mkldnn=use_mkldnn)
    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, acc, prediction


def lstm_net(data,
             label,
             dict_dim,
             emb_dim=128,
             hid_dim=128,
             hid_dim2=96,
             class_dim=2,
             emb_lr=30.0,
             use_mkldnn=False):
    """
    lstm net
    """
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(learning_rate=emb_lr))

    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 4, use_mkldnn=use_mkldnn)

    lstm_h, c = fluid.layers.dynamic_lstm(
        input=fc0, size=hid_dim * 4, is_reverse=False)

    lstm_max = fluid.layers.sequence_pool(input=lstm_h, pool_type='max')
    lstm_max_tanh = fluid.layers.tanh(lstm_max, use_mkldnn=use_mkldnn)

    fc1 = fluid.layers.fc(input=lstm_max_tanh, size=hid_dim2, act='tanh', use_mkldnn=use_mkldnn)

    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax', use_mkldnn=use_mkldnn)

    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, acc, prediction


def gru_net(data,
            label,
            dict_dim,
            emb_dim=128,
            hid_dim=128,
            hid_dim2=96,
            class_dim=2,
            emb_lr=400.0,
            use_mkldnn=False):
    """
    gru net
    """
    emb = fluid.layers.embedding(
        input=data,
        size=[dict_dim, emb_dim],
        param_attr=fluid.ParamAttr(learning_rate=emb_lr))

    fc0 = fluid.layers.fc(input=emb, size=hid_dim * 3, use_mkldnn=use_mkldnn)
    gru_h = fluid.layers.dynamic_gru(input=fc0, size=hid_dim, is_reverse=False)
    gru_max = fluid.layers.sequence_pool(input=gru_h, pool_type='max')
    gru_max_tanh = fluid.layers.tanh(gru_max, use_mkldnn=use_mkldnn)
    fc1 = fluid.layers.fc(input=gru_max_tanh, size=hid_dim2, act='tanh', use_mkldnn=use_mkldnn)
    prediction = fluid.layers.fc(input=fc1, size=class_dim, act='softmax', use_mkldnn=use_mkldnn)

    cost = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc = fluid.layers.accuracy(input=prediction, label=label)

    return avg_cost, acc, prediction

import sys
import time
import unittest
import contextlib
import argparse

import paddle.fluid as fluid

import utils
from nets import bow_net
from nets import cnn_net
from nets import lstm_net
from nets import bilstm_net
from nets import gru_net

nets = {'bow': bow_net, 'cnn': cnn_net, 'lstm': lstm_net, 'bilstm': bilstm_net,
        'gru': gru_net}
# learning rates
lrs = {'bow': 0.002, 'cnn': 0.01, 'lstm': 0.05, 'bilstm':0.002, 'gru': 0.05}

def parse_args():
    parser = argparse.ArgumentParser("Run inference.")
    parser.add_argument(
        'topology',
        type=str,
        choices=['bow', 'cnn', 'lstm', 'bilstm', 'gru'],
        help='Topology used for the model (bow/cnn/lstm/gru).')
    parser.add_argument(
        "--dataset",
        type=str,
        default='imdb',
        choices=['imdb', 'data'],
        help="Dataset to be used: 'imdb' or 'data' (from 'data' subdirectory).")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='The size of a batch. (default: %(default)d, usually: 128 for "bow" and "gru", 4 for "cnn", "lstm" and "bilstm").')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type. (default: %(default)s)')
    parser.add_argument(
        '--model_save_dir',
        type=str,
        default='./model',
        help='A directory for saving models. (default: %(default)s)')
    parser.add_argument(
        '--num_passes',
        type=int,
        default=30,
        help='The number of epochs. (default: %(default)d)')
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='If set, do calculation in parallel.')
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def train(train_reader,
          word_dict,
          network,
          use_cuda,
          parallel,
          save_dirname,
          lr=0.2,
          batch_size=128,
          pass_num=30):
    """
    train network
    """
    data = fluid.layers.data(
        name="words", shape=[1], dtype="int64", lod_level=1)

    label = fluid.layers.data(name="label", shape=[1], dtype="int64")

    if not parallel:
        cost, acc, prediction = network(data, label, len(word_dict))
    else:
        places = fluid.layers.get_places(device_count=2)
        pd = fluid.layers.ParallelDo(places)
        with pd.do():
            cost, acc, prediction = network(
                pd.read_input(data), pd.read_input(label), len(word_dict))

            pd.write_output(cost)
            pd.write_output(acc)

        cost, acc = pd()
        cost = fluid.layers.mean(cost)
        acc = fluid.layers.mean(acc)

    sgd_optimizer = fluid.optimizer.Adagrad(learning_rate=lr)
    sgd_optimizer.minimize(cost)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

    exe.run(fluid.default_startup_program())
    for pass_id in xrange(pass_num):
        data_size, data_count, total_acc, total_cost = 0, 0, 0.0, 0.0
        for data in train_reader():
            avg_cost_np, avg_acc_np = exe.run(fluid.default_main_program(),
                                              feed=feeder.feed(data),
                                              fetch_list=[cost, acc])
            data_size = len(data)
            total_acc += data_size * avg_acc_np
            total_cost += data_size * avg_cost_np
            data_count += data_size
        avg_cost = total_cost / data_count

        avg_acc = total_acc / data_count
        print("pass_id: %d, avg_acc: %f, avg_cost: %f" %
              (pass_id, avg_acc, avg_cost))

        epoch_model = save_dirname + "/" + "epoch" + str(pass_id)
        fluid.io.save_inference_model(epoch_model, ["words", "label"], acc, exe)


def train_net(args):
    word_dict, train_reader, test_reader = utils.prepare_data(
        args.dataset, self_dict=False, batch_size=128, buf_size=50000)

    train(
        train_reader,
        word_dict,
        nets[args.topology],
        use_cuda=(args.device == "GPU"),
        parallel=args.parallel,
        save_dirname=args.model_save_dir,
        lr=lrs[args.topology],
        pass_num=args.num_passes,
        batch_size=args.batch_size)


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    train_net(args)

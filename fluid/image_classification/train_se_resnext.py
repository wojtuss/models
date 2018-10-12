#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy as np
import time
import os

import cProfile

import paddle
import paddle.fluid as fluid
import paddle.fluid.core as core
import paddle.fluid.profiler as profiler

import reader
import models


def parse_args():
    parser = argparse.ArgumentParser('Convolution model benchmark.')
    parser.add_argument(
        '--model',
        type=str,
        choices=['SE_ResNeXt50_32x4d', 'SE_ResNeXt101_32x4d', 'SE_ResNeXt152_32x4d'],
        default='SE_ResNeXt50_32x4d',
        help='The model architecture.')
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        help='The minibatch size.')
    parser.add_argument(
        '--use_fake_data',
        action='store_true',
        help='use real data or fake data')
    parser.add_argument(
        '--skip_batch_num',
        type=int,
        default=0,
        help='The first num of minibatch num to skip, for better performance test.')
    parser.add_argument(
        '--iterations',
        type=int,
        default=3,
        help='The number of minibatches. 0 or less: whole dataset. Greater than 0: wraps the dataset up if necessary.')
    parser.add_argument(
        '--pass_num',
        type=int,
        default=1,
        help='The number of passes.')
    parser.add_argument(
        '--data_format',
        type=str,
        default='NCHW',
        choices=['NCHW', 'NHWC'],
        help='The data data_format, now only support NCHW.')
    parser.add_argument(
        '--device',
        type=str,
        default='CPU',
        choices=['CPU', 'GPU'],
        help='The device type.')
    parser.add_argument(
        '--data_set',
        type=str,
        default='imagenet',
        choices=['cifar10', 'flowers', 'imagenet'],
        help='Optional dataset for benchmark.')
    parser.add_argument(
        '--infer_only',
        action='store_true',
        help='If set, run forward only.')
    parser.add_argument(
        '--use_cprof',
        action='store_true',
        help='If set, use cProfile.')
    parser.add_argument(
        '--use_nvprof',
        action='store_true',
        help='If set, use nvprof for CUDA.')
    parser.add_argument(
        '--skip_test',
        action='store_true',
        help='If set, skip testing the model during training.')
    parser.add_argument(
        '--save_model',
        action='store_true',
        help='If set, save the model after each epoch.')
    parser.add_argument(
        '--save_model_path',
        type=str,
        default='output/SE_ResNeXt50_32x4d',
        help='A path for saving model.')
    parser.add_argument(
        '--train_file_list',
        type=str,
        default='/home/kbinias/data/imagenet/val_list.txt',
        help='A file with a list of training data files.')
    parser.add_argument(
        '--test_file_list',
        type=str,
        default='/home/kbinias/data/imagenet/val_list.txt',
        help='A file with a list of test data files.')
    parser.add_argument(
        '--data_dir',
        type=str,
        default='/home/kbinias/data/imagenet',
        help='A directory with train and test data files.')

    args = parser.parse_args()
    return args



def user_data_reader(data):
    """
    Creates a data reader whose data output is user data.
    """

    def data_reader():
        while True:
            for b in data:
                yield b

    return data_reader


def train(model, args):
    os.environ['FLAGS_use_mkldnn'] = "0"
    print("NOTICE:1. Even the globaly export FLAGS_use_mkldnn=true, the local FLAGS_use_mkldnn has been set False\n")
    print("NOTICE:2. Keep iteration_num and pass_num small. After finishing, copy the /output/SE_ResNeXt50_32x4d/__model__ file together with weights from baidu for inference\n")

    if args.use_cprof:
        pr = cProfile.Profile()
        pr.enable()

    if args.data_set == "cifar10":
        class_dim = 10
        if args.data_format == 'NCHW':
            dshape = [3, 32, 32]
        else:
            dshape = [32, 32, 3]
    elif args.data_set == "imagenet":
        class_dim = 1000
        if args.data_format == 'NCHW':
            dshape = [3, 224, 224]
        else:
            dshape = [224, 224, 3]
    else:
        class_dim = 102
        if args.data_format == 'NCHW':
            dshape = [3, 224, 224]
        else:
            dshape = [224, 224, 3]

    fake_train_data = [(np.random.rand(dshape[0] * dshape[1] * dshape[2]).
                        astype(np.float32), np.random.randint(1, class_dim))
                       for _ in range(1)]
    fake_test_data = [(np.random.rand(dshape[0] * dshape[1] * dshape[2]).
                       astype(np.float32), np.random.randint(1, class_dim))
                      for _ in range(1)]

    input = fluid.layers.data(name='data', shape=dshape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    predict = model(input, class_dim)
    cost = fluid.layers.cross_entropy(input=predict, label=label)
    avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=predict, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=predict, label=label, k=5)

    inference_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.Momentum(learning_rate=0.01, momentum=0.9)
    opts = optimizer.minimize(avg_cost)

    fluid.memory_optimize(fluid.default_main_program())

    # Prepare fake data
    if args.use_fake_data:
        train_reader = paddle.batch(
            user_data_reader(fake_train_data), batch_size=args.batch_size)
        test_reader =  paddle.batch(
            user_data_reader(fake_test_data), batch_size=args.batch_size)
    else:
        cycle = args.iterations > 0
        if args.data_set == 'cifar10':
            train_reader = paddle.batch(
                paddle.reader.shuffle(paddle.dataset.cifar.train10(cycle=cycle),
                                      buf_size=5120),
                batch_size=args.batch_size)
            test_reader = paddle.batch(paddle.dataset.cifar.test10(cycle=cycle),
                                       batch_size=args.batch_size)
        elif args.data_set == 'imagenet':
            train_reader = paddle.batch(
                reader.train(file_list=args.train_file_list,
                             data_dir=args.data_dir, cycle=cycle),
                batch_size=args.batch_size)
            test_reader = paddle.batch(
                reader.test(file_list=args.test_file_list,
                            data_dir=args.data_dir, cycle=cycle),
                batch_size=args.batch_size)
        else:
            train_reader = paddle.batch(
                paddle.reader.shuffle(paddle.dataset.flowers.train(cycle=cycle),
                                      buf_size=5120),
                batch_size=args.batch_size)
            test_reader = paddle.batch(paddle.dataset.flowers.test(cycle=cycle),
                                       batch_size=args.batch_size)

    def test(exe):
        for batch_id, data in enumerate(test_reader()):
            img_data = np.array(map(lambda x: x[0].reshape(dshape),
                                    data)).astype("float32")
            y_data = np.array(map(lambda x: x[1], data)).astype("int64")
            y_data = y_data.reshape([-1, 1])

            loss, acc1, acc5 = exe.run(inference_program,
                                       feed={"data": img_data,
                                             "label": y_data},
                                       fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])
            loss = np.mean(loss)
            acc1 = np.mean(acc1)
            acc5 = np.mean(acc5)

        return acc1

    # place = core.CPUPlace() if args.device == 'CPU' else core.CUDAPlace(0)
    if args.device == 'GPU':
        place = fluid.CUDAPlace(0)

    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())
    if args.use_fake_data:
        data = train_reader().next()
        image = np.array(map(lambda x: x[0].reshape(dshape), data)).astype(
            'float32')
        label = np.array(map(lambda x: x[1], data)).astype('int64')
        label = label.reshape([-1, 1])

    for pass_id in range(args.pass_num):
        train_accs = []
        train_losses = []
        batch_times = []
        fpses = []
        iters = 0
        total_samples = 0
        train_start_time = time.time()
        for data in train_reader():
            if args.iterations > 0 and iters == args.iterations + args.skip_batch_num:
                break
            if iters == args.skip_batch_num:
                profiler.reset_profiler()
                total_samples = 0
                train_start_time = time.time()
            if not args.use_fake_data:
                image = np.array(map(lambda x: x[0].reshape(dshape),
                                     data)).astype('float32')
                label = np.array(map(lambda x: x[1], data)).astype('int64')
                label = label.reshape([-1, 1])

            start = time.time()
            loss, acc1, acc5 = exe.run(
                fluid.default_main_program(),
                feed={'data': image,
                      'label': label},
                fetch_list=[avg_cost.name, acc_top1.name, acc_top5.name])
            loss = np.mean(loss)
            acc1 = np.mean(acc1)
            acc5 = np.mean(acc5)
            batch_time = time.time() - start
            samples = len(label)
            total_samples += samples
            fps = samples / batch_time
            iters += 1
            train_losses.append(loss)
            train_accs.append(acc1)
            batch_times.append(batch_time)
            fpses.append(fps)
            appx = ' (warm-up)' if iters <= args.skip_batch_num else ''
            print("Pass: %d, Iter: %d%s, Loss: %f, Accuracy: %f, FPS: %.5f img/s" %
                  (pass_id, iters, appx, loss, acc1, fps))

            # Saving model every iteration to generate __model__
            if args.save_model:
                if not os.path.isdir(args.save_model_path):
                    os.makedirs(args.save_model_path)
                fluid.io.save_inference_model(args.save_model_path,
                                              ["data", "label"], [avg_cost, acc_top1, acc_top5], exe)
                print("Model saved into {}".format(args.save_model_path))

            # evaluation
            #  if not args.skip_test:

        #  pass_test_acc = test(exe)

        # Postprocess benchmark data
        latencies = batch_times[args.skip_batch_num:]
        latency_avg = np.average(latencies)
        latency_std = np.std(latencies)
        latency_pc99 = np.percentile(latencies, 99)
        fps_avg = np.average(fpses)
        fps_std = np.std(fpses)
        fps_pc01 = np.percentile(fpses, 1)
        train_total_time = time.time() - train_start_time
        examples_per_sec = total_samples / train_total_time

        # Benchmark output
        print("\nPass %d statistics:" % (pass_id))
        print("Loss: %f, Train Accuracy: %f" %
              (np.mean(train_losses), np.mean(train_accs)))
        print('Avg fps: %.5f, std fps: %.5f, fps for 99pc latency: %.5f' %
              (fps_avg, fps_std, fps_pc01))
        print('Avg latency: %.5f, std latency: %.5f, 99pc latency: %.5f' %
              (latency_avg, latency_std, latency_pc99))
        print('Total examples: %d, total time: %.5f, total examples/sec: %.5f\n' %
              (total_samples, train_total_time, examples_per_sec))

        # Save model
        if args.save_model:
            if not os.path.isdir(args.save_model_path):
                os.makedirs(args.save_model_path)
            fluid.io.save_inference_model(args.save_model_path,
                                          ["data", "label"], [avg_cost, acc_top1, acc_top5], exe)
            print("Model saved into {}".format(args.save_model_path))

        # evaluation
        #  if not args.skip_test:
        #  pass_test_acc = test(exe)


def SE_ResNeXt50_32x4d(input, class_dim):
    SE_ResNeXt50_32x4d_model = models.__dict__["SE_ResNeXt50_32x4d"]()
    return SE_ResNeXt50_32x4d_model.net(input, class_dim)


def print_arguments(args):
    vars(args)['use_nvprof'] = (vars(args)['use_nvprof'] and
                                vars(args)['device'] == 'GPU')
    print('----------- resnet Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


if __name__ == '__main__':
    model_map = {
        #which args model name, and which exact function of this model
        'SE_ResNeXt50_32x4d': SE_ResNeXt50_32x4d
    }
    args = parse_args()
    print_arguments(args)
    if args.data_format == 'NHWC':
        raise ValueError('Only support NCHW data_format now.')
    if args.use_nvprof and args.device == 'GPU':
        with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
            train(model_map[args.model], args)
    else:
        with profiler.profiler(args.device, sorted_key = 'total') as cpuprof:
            train(model_map[args.model], args)

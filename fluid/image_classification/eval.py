import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import models
import reader
import argparse
import functools
from models.learning_rate import cosine_decay
from utility import add_arguments, print_arguments
import math

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  256,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('class_dim',        int,  1000,                "Class number.")
add_arg('image_shape',      str,  "3,224,224",         "Input image size")
add_arg('with_mem_opt',     bool, True,                "Whether to use memory optimization or not.")
add_arg('pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('model',            str, "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('iterations',       int,  0,                   "The number of iterations. Zero or less means whole test set. More than 0 means the training set might be looped until # of iterations is reached.")
add_arg('skip_batch_num',   int,  0,                   "The first num of minibatch num to skip, for better performance test.")
add_arg('profile',          bool, False,               "If set, do profiling.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def eval(args):
    # parameters from arguments
    class_dim = args.class_dim
    model_name = args.model
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # model definition
    model = models.__dict__[model_name]()

    if model_name is "GoogleNet":
        out0, out1, out2 = model.net(input=image, class_dim=class_dim)
        cost0 = fluid.layers.cross_entropy(input=out0, label=label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=label)
        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)

        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=label, k=5)
    else:
        out = model.net(input=image, class_dim=class_dim)
        cost = fluid.layers.cross_entropy(input=out, label=label)

        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

    test_program = fluid.default_main_program().clone(for_test=True)

    if with_memory_optimization:
        fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    val_reader = paddle.batch(reader.test(cycle=args.iterations>0), batch_size=args.batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]

    test_info = [[], [], []]
    cnt = 0
    batch_times = []
    for batch_id, data in enumerate(val_reader()):
        iters = batch_id
        if args.iterations == args.skip_batch_num:
            profiler.reset_profiler()
        elif batch_id < args.skip_batch_num:
            print("Warm-up iteration")
        if args.iterations > 0 and batch_id == args.iterations + args.skip_batch_num:
            break
        t1 = time.time()
        loss, acc1, acc5 = exe.run(test_program,
                                   fetch_list=fetch_list,
                                   feed=feeder.feed(data))
        t2 = time.time()
        period = t2 - t1
        fps = args.batch_size / period
        batch_times.append(period)
        loss = np.mean(loss)
        acc1 = np.mean(acc1)
        acc5 = np.mean(acc5)
        test_info[0].append(loss * len(data))
        test_info[1].append(acc1 * len(data))
        test_info[2].append(acc5 * len(data))
        cnt += len(data)
        if batch_id % 10 == 0:
            print("Testbatch {0},loss {1}, "
                  "acc1 {2},acc5 {3},time {4}, fps {5}".format(batch_id, \
                  loss, acc1, acc5, \
                  "%2.2f sec" % period, fps))
            sys.stdout.flush()

    test_loss = np.sum(test_info[0]) / cnt
    test_acc1 = np.sum(test_info[1]) / cnt
    test_acc5 = np.sum(test_info[2]) / cnt

    print("Test_loss {0}, test_acc1 {1}, test_acc5 {2}".format(
        test_loss, test_acc1, test_acc5))
    sys.stdout.flush()

    latencies = batch_times[args.skip_batch_num:]
    latency_avg = np.average(latencies)
    latency_pc99 = np.percentile(latencies, 99)
    fpses = np.divide(args.batch_size, latencies)
    fps_avg = np.average(fpses)
    fps_pc99 = np.percentile(fpses, 1)

    # Benchmark output
    print('\nTotal examples (incl. warm-up): %d' % (iters * args.batch_size))
    print('average latency: %.5f s, 99pc latency: %.5f s' % (latency_avg,
                                                             latency_pc99))
    print('average fps: %.5f, fps for 99pc latency: %.5f' % (fps_avg, fps_pc99))


def main():
    args = parser.parse_args()
    print_arguments(args)
    if args.profile:
        if args.use_gpu:
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                eval(args)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                eval(args)
    else:
        eval(args)


if __name__ == '__main__':
    main()

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
add_arg('batch_size',       int,   256,                  "Minibatch size.")
add_arg('use_gpu',          bool,  True,                 "Whether to use GPU or not.")
add_arg('total_images',     int,   1281167,              "Training image number.")
add_arg('num_epochs',       int,   120,                  "number of epochs.")
add_arg('class_dim',        int,   1000,                 "Class number.")
add_arg('image_shape',      str,   "3,224,224",          "input image size")
add_arg('model_save_dir',   str,   "output",             "model save directory")
add_arg('with_mem_opt',     bool,  True,                 "Whether to use memory optimization or not.")
add_arg('pretrained_model', str,   None,                 "Whether to use pretrained model.")
add_arg('checkpoint',       str,   None,                 "Whether to resume checkpoint.")
add_arg('lr',               float, 0.1,                  "set learning rate.")
add_arg('lr_strategy',      str,   "piecewise_decay",    "Set the learning rate decay strategy.")
add_arg('model',            str,   "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('iterations',       int,   0,                    "The number of iterations. Zero or less means whole training set. More than 0 means the training set might be looped until # of iterations is reached.")
add_arg('skip_test',        bool,  True,                 "Whether to skip test phase.")
add_arg('profile',          bool,  False,                "If set, do profiling.")
add_arg('skip_batch_num',   int,   0,                    "The number of first minibatches to skip as warm-up for better performance test.")
add_arg('use_fake_data',    int,   0,                    "Use real data or fake data")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def optimizer_setting(params):
    ls = params["learning_strategy"]

    if ls["name"] == "piecewise_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]

        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        bd = [step * e for e in ls["epochs"]]
        base_lr = params["lr"]
        lr = []
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    elif ls["name"] == "cosine_decay":
        if "total_images" not in params:
            total_images = 1281167
        else:
            total_images = params["total_images"]

        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)

        lr = params["lr"]
        num_epochs = params["num_epochs"]

        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))
    else:
        lr = params["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=0.9,
            regularization=fluid.regularizer.L2Decay(1e-4))

    return optimizer

def user_data_reader(data):
    """
    Creates a data reader whose data output is user data.
    """

    def data_reader():
        while True:
            for b in data:
                yield b

    return data_reader

def train(args):
    # parameters from arguments
    class_dim = args.class_dim
    model_name = args.model
    checkpoint = args.checkpoint
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    model_save_dir = args.model_save_dir
    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    if args.use_fake_data:
        fake_train_data = [(np.random.rand(image_shape[0] * image_shape[1] * image_shape[2]).
                            astype(np.float32), np.random.randint(1, class_dim))
                           for _ in range(1)]
        fake_test_data = [(np.random.rand(image_shape[0] * image_shape[1] * image_shape[2]).
                           astype(np.float32), np.random.randint(1, class_dim))
                          for _ in range(1)]

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

    # parameters from model and arguments
    params = model.params
    params["total_images"] = args.total_images
    params["lr"] = args.lr
    params["num_epochs"] = args.num_epochs
    params["learning_strategy"]["batch_size"] = args.batch_size
    params["learning_strategy"]["name"] = args.lr_strategy

    # initialize optimizer
    optimizer = optimizer_setting(params)
    opts = optimizer.minimize(avg_cost)

    if with_memory_optimization:
        fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if checkpoint is not None:
        fluid.io.load_persistables(exe, checkpoint)

    if pretrained_model:

        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    train_batch_size = args.batch_size
    test_batch_size = 16
    if args.use_fake_data:
        train_reader = paddle.batch(
            user_data_reader(fake_train_data), batch_size=args.batch_size)
        test_reader = paddle.batch(
            user_data_reader(fake_test_data), batch_size=test_batch_size)
        feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    else:
        train_reader = paddle.batch(
            reader.train(cycle=args.iterations > 0), batch_size=train_batch_size)
        test_reader = paddle.batch(reader.test(), batch_size=test_batch_size)
        feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    train_exe = fluid.ParallelExecutor(
        use_cuda=True if args.use_gpu else False, loss_name=avg_cost.name)

    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]

    iters = 0
    for pass_id in range(params["num_epochs"]):
        iters = pass_id
        train_info = [[], [], []]
        test_info = [[], [], []]
        batch_times = []
        for batch_id, data in enumerate(train_reader()):
            if batch_id == args.skip_batch_num:
                profiler.reset_profiler()
            elif batch_id < args.skip_batch_num:
                print("Warm-up iteration")
            if args.iterations > 0 and batch_id == args.iterations + args.skip_batch_num:
                break
            t1 = time.time()
            loss, acc1, acc5 = train_exe.run(fetch_list, feed=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            batch_times.append(period)
            fps = args.batch_size / period
            loss = np.mean(np.array(loss))
            acc1 = np.mean(np.array(acc1))
            acc5 = np.mean(np.array(acc5))
            train_info[0].append(loss)
            train_info[1].append(acc1)
            train_info[2].append(acc5)
            if batch_id % 10 == 0:
                print("Train pass {0}, trainbatch {1}, loss {2}, \
                        acc1 {3}, acc5 {4}, latency {5}, fps {6}"
                                                   .format(pass_id, \
                       batch_id, loss, acc1, acc5, \
                       "%2.2f sec" % period, fps))
                sys.stdout.flush()

        train_loss = np.array(train_info[0]).mean()
        train_acc1 = np.array(train_info[1]).mean()
        train_acc5 = np.array(train_info[2]).mean()

        if not args.skip_test:
            cnt = 0
            for test_batch_id, data in enumerate(test_reader()):
                t1 = time.time()
                loss, acc1, acc5 = exe.run(test_program,
                                           fetch_list=fetch_list,
                                           feed=feeder.feed(data))
                t2 = time.time()
                period = t2 - t1
                loss = np.mean(loss)
                acc1 = np.mean(acc1)
                acc5 = np.mean(acc5)
                test_info[0].append(loss * len(data))
                test_info[1].append(acc1 * len(data))
                test_info[2].append(acc5 * len(data))
                cnt += len(data)
                if test_batch_id % 10 == 0:
                    print("Pass {0},testbatch {1},loss {2}, \
                        acc1 {3},acc5 {4},time {5}"
                                                    .format(pass_id, \
                        test_batch_id, loss, acc1, acc5, \
                        "%2.2f sec" % period))
                    sys.stdout.flush()

            test_loss = np.sum(test_info[0]) / cnt
            test_acc1 = np.sum(test_info[1]) / cnt
            test_acc5 = np.sum(test_info[2]) / cnt

            print("End pass {0}, train_loss {1}, train_acc1 {2}, train_acc5 {3}, "
                "test_loss {4}, test_acc1 {5}, test_acc5 {6}".format(pass_id, \
                train_loss, train_acc1, train_acc5, test_loss, test_acc1, \
                test_acc5))
            sys.stdout.flush()

        model_path = os.path.join(model_save_dir + '/' + model_name,
                                  str(pass_id))
        if not os.path.isdir(model_path):
            os.makedirs(model_path)
        fluid.io.save_persistables(exe, model_path)

        latencies = batch_times[args.skip_batch_num:]
        latency_avg = np.average(latencies)
        latency_pc99 = np.percentile(latencies, 99)
        fpses = np.divide(args.batch_size, latencies)
        fps_avg = np.average(fpses)
        fps_pc99 = np.percentile(fpses, 1)

        # Benchmark output
        print('\nTotal examples (incl. warm-up): %d' %
              (iters * args.batch_size))
        print('average latency: %.5f s, 99pc latency: %.5f s' % (latency_avg,
                                                                 latency_pc99))
        print('average fps: %.5f, fps for 99pc latency: %.5f' % (fps_avg,
                                                                 fps_pc99))


def main():
    args = parser.parse_args()
    print_arguments(args)
    if args.profile:
        if args.use_gpu:
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                train(args)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                train(args)
    else:
        train(args)


if __name__ == '__main__':
    main()

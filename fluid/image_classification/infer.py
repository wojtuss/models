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
# yapf: disable
add_arg = functools.partial(add_arguments, argparser=parser)
add_arg('batch_size',        int,   256,            "Minibatch size.")
add_arg('use_gpu',           bool,  True,           "Whether to use GPU or not.")
add_arg('class_dim',         int,   1000,           "Class number.")
add_arg('image_shape',       str,   "3,224,224",    "Input image size")
add_arg('with_mem_opt',      bool,  True,           "Whether to use memory optimization or not.")
add_arg('pretrained_model',  str,   None,           "Whether to use pretrained model.")
add_arg('model',             str,                   "SE_ResNeXt50_32x4d", "Set the network to use.")
add_arg('profile',           bool,  False,          "If set, do profiling.")
add_arg('iterations',        int,   0,              "The number of iterations. Zero or less means whole training set. More than 0 means the training set might be looped until # of iterations is reached.")
add_arg('use_fake_data',     bool,  False,          "Use real data or fake data")
add_arg('skip_batch_num',    int,   0,              "The number of first minibatches to skip as warm-up for better performance test.")
add_arg('skip_test',         bool,  True,           "Whether to skip test phase.")
add_arg('num_epochs',        int,   120,            "number of epochs.")
# use_transpiler must be setting False, because of failing when True
add_arg('use_transpiler',    bool,  False,          "Whether to use transpiler.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]


def user_data_reader(data):
    """
    Creates a data reader whose data output is user data.
    """

    def data_reader():
        while True:
            for b in data:
                yield b

    return data_reader


def infer(args):
    # parameters from arguments
    class_dim = args.class_dim
    model_name = args.model
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    if args.use_fake_data:
        fake_test_data = [(np.random.rand(image_shape[0] * image_shape[1] * image_shape[2]).
                           astype(np.float32), np.random.randint(1, class_dim))
                          for _ in range(1)]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # model definition
    model = models.__dict__[model_name]()

    if model_name is "GoogleNet":
        out, _, _ = model.net(input=image, class_dim=class_dim)
    else:
        out = model.net(input=image, class_dim=class_dim)

    test_program = fluid.default_main_program().clone(for_test=True)

    if with_memory_optimization:
        fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if args.use_transpiler:
        inference_transpiler_program = test_program
        t = fluid.InferenceTranspiler()
        t.transpile(inference_transpiler_program, place)
        test_program = inference_transpiler_program

    # parameters from model and arguments
    params = model.params
    params["num_epochs"] = args.num_epochs

    if pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)

    test_batch_size = 1
    if args.use_fake_data:
        test_reader = paddle.batch(
            user_data_reader(fake_test_data), batch_size=args.batch_size)
        feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    else:
        test_reader = paddle.batch(reader.test(cycle=args.iterations > 0), batch_size=args.batch_size)
        feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    fetch_list = [out.name]

    TOPK = 1
    iters = 0
    for pass_id in range(params["num_epochs"]):
        iters = pass_id
        batch_times = []
        for batch_id, data in enumerate(test_reader()):
            if batch_id == args.skip_batch_num:
                profiler.reset_profiler()
            elif batch_id < args.skip_batch_num:
                print("Warm-up iteration")
            if args.iterations > 0 and batch_id == args.iterations + args.skip_batch_num:
                break
            t1 = time.time()
            result = exe.run(test_program,
                             fetch_list=fetch_list,
                             feed=feeder.feed(data))
            t2 = time.time()
            period = t2 - t1
            result = result[0][0]
            pred_label = np.argsort(result)[::-1][:TOPK]
            batch_times.append(period)
            fps = args.batch_size / period
            if batch_id % 10 == 0:
                print("Test-{0}-score: {1}, class {2}, latency {3}, fps {4} img/s"
                      .format(batch_id, result[pred_label], pred_label, "%.2f sec" % period, fps))
            sys.stdout.flush()

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
                infer(args)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                infer(args)
    else:
        infer(args)


if __name__ == '__main__':
    main()

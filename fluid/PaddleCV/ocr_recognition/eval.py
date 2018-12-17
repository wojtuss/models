import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
from utility import add_arguments, print_arguments, to_lodtensor, get_ctc_feeder_data, get_attention_feeder_data
from attention_model import attention_eval
from crnn_ctc_model import ctc_eval
import data_reader
import argparse
import functools
import os
import time
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('model',             str,  "crnn_ctc", "Which type of network to be used. 'crnn_ctc' or 'attention'")
add_arg('model_path',        str,  "",         "The model path to be used for inference.")
add_arg('input_images_dir',  str,  None,       "The directory of images.")
add_arg('input_images_list', str,  None,       "The list file of images.")
add_arg('use_gpu',           bool, True,       "Whether use GPU to eval.")
add_arg('batch_size',        int,  1,          "Minibatch size.")
add_arg('use_transpiler',    bool, True,       "Whether to use transpiler.")
add_arg('iterations',        int, 0,           "The number of iterations. Zero or less means whole training set. More than 0 means the training set might be looped until # of iterations is reached.")
add_arg('profile',            bool, False,  "Whether to use profiling.")
add_arg('skip_batch_num',    int,   0,         "The number of first minibatches to skip as warm-up for better performance test.")
# yapf: enable


def evaluate(args):
    """OCR inference"""

    if args.model == "crnn_ctc":
        eval = ctc_eval
        get_feeder_data = get_ctc_feeder_data
    else:
        eval = attention_eval
        get_feeder_data = get_attention_feeder_data

    num_classes = data_reader.num_classes()
    data_shape = data_reader.data_shape()
    # define network
    decoded_out, evaluator = eval(data_shape, num_classes,
                                  True if args.use_gpu else False)

    # data reader
    test_reader = data_reader.test(
        batch_size=args.batch_size,
        cycle=args.iterations > 0,
        test_images_dir=args.input_images_dir,
        test_list_file=args.input_images_list,
        model=args.model)

    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load init model
    model_dir = args.model_path
    model_file_name = None
    if not os.path.isdir(args.model_path):
        model_dir = os.path.dirname(args.model_path)
        model_file_name = os.path.basename(args.model_path)
    fluid.io.load_params(exe, dirname=model_dir, filename=model_file_name)
    print("Init model from: %s." % args.model_path)

    program = fluid.default_main_program()
    if args.use_transpiler:
        inference_transpiler_program = program.clone()
        t = fluid.transpiler.InferenceTranspiler()
        t.transpile(inference_transpiler_program, place)
        program = inference_transpiler_program

    batch_times = []
    evaluator.reset(exe)
    count = 0
    for data in test_reader():
        if args.iterations > 0 and count == args.iterations + args.skip_batch_num:
            break
        if count < args.skip_batch_num:
            print("Warm-up itaration")
        if count == args.skip_batch_num:
            profiler.reset_profiler()

        start = time.time()
        exe.run(program, feed=get_feeder_data(data, place))

        batch_time = time.time() - start
        fps = args.batch_size / batch_time
        batch_times.append(batch_time)

        count += 1

    avg_distance, avg_seq_error = evaluator.eval(exe)
    print("Read %d samples; avg_distance: %s; avg_seq_error: %s" %
          (count, avg_distance, avg_seq_error))

    latencies = batch_times[args.skip_batch_num:]
    latency_avg = np.average(latencies)
    latency_pc99 = np.percentile(latencies, 99)
    fpses = np.divide(args.batch_size, latencies)
    fps_avg = np.average(fpses)
    fps_pc99 = np.percentile(fpses, 1)

    # Benchmark output
    print('\nTotal examples (incl. warm-up): %d' % (args.iterations * args.batch_size))
    print('average latency: %.5f s, 99pc latency: %.5f s' % (latency_avg,
                                                             latency_pc99))
    print('average fps: %.5f, fps for 99pc latency: %.5f' % (fps_avg, fps_pc99))

def main():
    args = parser.parse_args()
    print_arguments(args)
    if args.profile:
        if args.use_gpu:
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                evaluate(args)
        else:
            with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                evaluate(args)
    else:
        evaluate(args)


if __name__ == "__main__":
    main()

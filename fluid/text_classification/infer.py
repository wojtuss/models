import sys
import time
import unittest
import contextlib
import numpy as np
import argparse

import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import paddle.v2 as paddle

import utils


def parse_args():
    parser = argparse.ArgumentParser("Run inference.")
    parser.add_argument(
        '--batch_size',
        type=int,
        default=128,
        help='The size of a batch. (default: %(default)d, usually: 128 for "bow" and "gru", 4 for "cnn" and "lstm")')
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        choices=['CPU', 'GPU'],
        help='The device type. (default: %(default)s)')
    parser.add_argument(
        '--model_path',
        type=str,
        help='A directory with saved model.')
    parser.add_argument(
        '--skip_pass_num',
        type=int,
        default=0,
        help='The first num of passes to skip in statistics calculations.')
    parser.add_argument(
        '--iterations',
        type=int,
        default=0,
        help='The number of minibatches. (default: 0, i.e. all)')
    parser.add_argument(
        '--num_passes',
        type=int,
        default=30,
        help='The number of epochs. (default: %(default)d)')
    parser.add_argument(
        '--profile',
        action='store_true',
        help='If set, do profiling.')
    args = parser.parse_args()
    return args


def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def infer(args):
    """
    inference function
    """
    place = fluid.CUDAPlace(0) if args.device == 'GPU' else fluid.CPUPlace()
    exe = fluid.Executor(place)

    inference_scope = fluid.core.Scope()
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names,
         fetch_targets] = fluid.io.load_inference_model(args.model_path, exe)

        total_passes = args.num_passes + args.skip_pass_num
        batch_times = [0] * total_passes
        word_counts = [0] * total_passes
        wpses = [0] * total_passes
        word_dict, train_reader, test_reader = utils.prepare_data(
            "imdb", self_dict=False, batch_size=args.batch_size,
            buf_size=50000)
        total_acc = 0.0
        total_count = 0
        all_iters = 0
        for pass_id in range(total_passes):
            if pass_id < args.skip_pass_num:
                print("Warm-up pass.")
            if pass_id == args.skip_pass_num:
                profiler.reset_profiler()
            iters = 0
            for data in test_reader():
                if args.iterations and iters == args.iterations:
                    break
                start = time.time()
                acc = exe.run(inference_program,
                            feed=utils.data2tensor(data, place),
                            fetch_list=fetch_targets,
                            return_numpy=True)
                batch_time = time.time() - start
                word_count = len([w for d in data for w in d[0]])
                #  print("iter: {}, word_count: {}".format(iters, word_count))
                batch_times[pass_id] += batch_time
                word_counts[pass_id] += word_count
                iters += 1
                all_iters += 1
                total_acc += acc[0] * len(data)
                total_count += len(data)

            batch_times[pass_id] /= iters
            word_counts[pass_id] /= iters
            wps = word_counts[pass_id] / batch_times[pass_id]
            wpses[pass_id] = wps
            print("Pass: %d, iterations (total): %d (%d), latency: %.5f s, words: %d, wps: %f" %
                  (pass_id, iters, all_iters, batch_times[pass_id], word_counts[pass_id], wps))

            #  avg_acc = total_acc / total_count
            #  print("model_path: %s, avg_acc: %f" % (args.model_path, avg_acc))
            #  print("iters {}".format(iters))

    # Postprocess benchmark data
    latencies = batch_times[args.skip_pass_num:]
    latency_avg = np.average(latencies)
    latency_std = np.std(latencies)
    latency_pc99 = np.percentile(latencies, 99)
    wps_avg = np.average(wpses)
    wps_std = np.std(wpses)
    wps_pc01 = np.percentile(wpses, 1)

    # Benchmark output
    print('\nTotal passes (incl. warm-up): %d' % (total_passes))
    print('Total iterations (incl. warm-up): %d' % (all_iters))
    print('Total examples (incl. warm-up): %d' % (all_iters * args.batch_size))
    print('avg latency: %.5f, std latency: %.5f, 99pc latency: %.5f' %
          (latency_avg, latency_std, latency_pc99))
    print('avg wps: %.5f, std wps: %.5f, wps for 99pc latency: %.5f' %
          (wps_avg, wps_std, wps_pc01))


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)

    if args.profile:
        if args.device == 'GPU':
            with profiler.cuda_profiler("cuda_profiler_out.csv", 'csv') as nvprof:
                infer(args)
        else:
            with profiler.profiler('CPU', sorted_key='total') as cpuprof:
                infer(args)
    else:
        infer(args)


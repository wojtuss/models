import os
import time
import numpy as np
import argparse
import functools

import paddle
import paddle.fluid as fluid
import paddle.fluid.profiler as profiler
import reader
from mobilenet_ssd import mobile_net
from utility import add_arguments, print_arguments

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('dataset',          str,   'pascalvoc',  "coco2014, coco2017, and pascalvoc.")
add_arg('batch_size',       int,   32,        "Minibatch size.")
add_arg('use_gpu',          bool,  True,       "Whether to use GPU.")
add_arg('iterations',       int,   0,           "Number of batches to process (0 means all).")
add_arg('skip_batch_num',   int,   0,           "The first N batches to skip in statistics calculation. (default: 0)")
add_arg('data_dir',         str,   '',        "The data root path.")
add_arg('test_list',        str,   '',        "The testing data lists.")
add_arg('model_dir',        str,   '',     "The model path.")
add_arg('nms_threshold',    float, 0.45,   "NMS threshold.")
add_arg('ap_version',       str,   '11point',   "integral, 11point.")
add_arg('resize_h',         int,   300,    "The resized image height.")
add_arg('resize_w',         int,   300,    "The resized image height.")
add_arg('mean_value_B',     float, 127.5,  "Mean value for B channel which will be subtracted.")  #123.68
add_arg('mean_value_G',     float, 127.5,  "Mean value for G channel which will be subtracted.")  #116.78
add_arg('mean_value_R',     float, 127.5,  "Mean value for R channel which will be subtracted.")  #103.94
parser.add_argument('--use_mkldnn', action='store_true', help='If set, use MKL-DNN library.')
parser.add_argument('--profile', action='store_true', help='If set, do profiling.')
# yapf: enable


def eval(args, data_args, test_list):
    image_shape = [3, data_args.resize_h, data_args.resize_w]
    if 'coco' in data_args.dataset:
        num_classes = 91
    elif 'pascalvoc' in data_args.dataset:
        num_classes = 21

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    gt_box = fluid.layers.data(
        name='gt_box', shape=[4], dtype='float32', lod_level=1)
    gt_label = fluid.layers.data(
        name='gt_label', shape=[1], dtype='int32', lod_level=1)
    difficult = fluid.layers.data(
        name='gt_difficult', shape=[1], dtype='int32', lod_level=1)

    use_cudnn = True if args.use_gpu else False

    locs, confs, box, box_var = mobile_net(num_classes, image, image_shape,
                                           use_cudnn=use_cudnn,
                                           use_mkldnn=args.use_mkldnn)
    nmsed_out = fluid.layers.detection_output(
        locs, confs, box, box_var, nms_threshold=args.nms_threshold)
    loss = fluid.layers.ssd_loss(locs, confs, gt_box, gt_label, box, box_var)
    loss = fluid.layers.reduce_sum(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    # yapf: disable
    if args.model_dir:
        def if_exist(var):
            return os.path.exists(os.path.join(args.model_dir, var.name))
        fluid.io.load_vars(exe, args.model_dir, predicate=if_exist)
    # yapf: enable
    test_reader = paddle.batch(
        reader.test(data_args, test_list), batch_size=args.batch_size)
    feeder = fluid.DataFeeder(
        place=place, feed_list=[image, gt_box, gt_label, difficult])

    def test():
        # switch network to test mode (i.e. batch norm test mode)
        test_program = fluid.default_main_program().clone(for_test=True)
        with fluid.program_guard(test_program):
            map_eval = fluid.evaluator.DetectionMAP(
                nmsed_out,
                gt_label,
                gt_box,
                difficult,
                num_classes,
                overlap_threshold=0.5,
                evaluate_difficult=False,
                ap_version=args.ap_version)

        _, accum_map = map_eval.get_map_var()
        map_eval.reset(exe)
        batch_times = []
        iters = 0
        start = time.time()
        for batch_id, data in enumerate(test_reader()):
            if args.iterations and iters == args.iterations + args.skip_batch_num:
                break
            if iters == args.skip_batch_num:
                profiler.reset_profiler()
            test_map = exe.run(test_program,
                               feed=feeder.feed(data),
                               fetch_list=[accum_map])
            batch_time = time.time() - start
            start = time.time()
            fps = args.batch_size / batch_time
            batch_times.append(batch_time)
            iters += 1
            appx = ' (warm-up)' if iters <= args.skip_batch_num else ''
            print("Batch: %d%s, latency: %.5f s, fps: %f" %
                  (iters, appx, batch_time, fps))

        # Postprocess benchmark data
        latencies = batch_times[args.skip_batch_num:]
        latency_avg = np.average(latencies)
        latency_std = np.std(latencies)
        latency_pc99 = np.percentile(latencies, 99)
        fpses = np.divide(args.batch_size, latencies)
        fps_avg = np.average(fpses)
        fps_std = np.std(fpses)
        fps_pc99 = np.percentile(fpses, 1)

        # Benchmark output
        print('\nTotal images (incl. warm-up): %d' % (iters * args.batch_size))
        print('avg latency: %.5f s, std latency: %.5f s, 99pc latency: %.5f s' %
                (latency_avg, latency_std, latency_pc99))
        print('avg fps: %.5f, std fps: %.5f, fps for 99pc latency: %.5f' %
                (fps_avg, fps_std, fps_pc99))

    test()


if __name__ == '__main__':
    args = parser.parse_args()
    print_arguments(args)

    data_dir = 'data/pascalvoc'
    test_list = 'test.txt'
    label_file = 'label_list'

    if not os.path.exists(args.model_dir):
        raise ValueError("The model path [%s] does not exist." %
                         (args.model_dir))
    if 'coco' in args.dataset:
        data_dir = 'data/coco'
        if '2014' in args.dataset:
            test_list = 'annotations/instances_val2014.json'
        elif '2017' in args.dataset:
            test_list = 'annotations/instances_val2017.json'

    data_args = reader.Settings(
        dataset=args.dataset,
        data_dir=args.data_dir if len(args.data_dir) > 0 else data_dir,
        label_file=label_file,
        resize_h=args.resize_h,
        resize_w=args.resize_w,
        mean_value=[args.mean_value_B, args.mean_value_G, args.mean_value_R],
        apply_distort=False,
        apply_expand=False,
        ap_version=args.ap_version,
        toy=0)

    if args.profile:
        if args.use_gpu:
            with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                eval(args, data_args=data_args,
                    test_list=args.test_list if len(args.test_list) > 0
                     else test_list)
        else:
            with profiler.profiler('CPU', sorted_key='total') as cpuprof:
                eval(args, data_args=data_args,
                    test_list=args.test_list if len(args.test_list) > 0
                     else test_list)
    else:
        eval(args, data_args=data_args,
            test_list=args.test_list if len(args.test_list) > 0
             else test_list)


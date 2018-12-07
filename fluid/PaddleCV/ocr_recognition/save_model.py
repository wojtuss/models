"""
To save a CRNN-CTC model run something like:
    LD_LIBRARY_PATH=a/path/to/Paddle/build/third_party/install/warpctc/lib/ python save_model.py --save_model_dir=my_model
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import paddle.fluid as fluid
from utility import add_arguments, print_arguments, to_lodtensor, get_ctc_feeder_data, get_attention_feeder_data
import paddle.fluid.profiler as profiler
from crnn_ctc_model import ctc_eval, ctc_eval_noacc
from attention_model import attention_train_net
import data_reader
import argparse
import functools
import sys
import time
import os
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',        int,   1,         "Minibatch size.")
add_arg('total_step',        int,   7,    "The number of iterations. Zero or less means whole training set. More than 0 means the training set might be looped until # of iterations is reached.")
add_arg('log_period',        int,   1,       "Log period.")
add_arg('save_model_period', int,   1,      "Save model period. '-1' means never saving the model.")
add_arg('eval_period',       int,   1,      "Evaluate period. '-1' means never evaluating the model.")
add_arg('save_model_dir',    str,   "./models", "The directory the model to be saved to.")
add_arg('train_images',      str,   None,       "The directory of images to be used for training.")
add_arg('train_list',        str,   None,       "The list file of images to be used for training.")
add_arg('test_images',       str,   None,       "The directory of images to be used for test.")
add_arg('test_list',         str,   None,       "The list file of images to be used for training.")
add_arg('model',    str,   "crnn_ctc",           "Which type of network to be used. 'crnn_ctc' or 'attention'")
add_arg('init_model',        str,   None,       "The init model file of directory.")
add_arg('use_gpu',           bool,  False,      "Whether use GPU to train.")
add_arg('min_average_window',int,   1,     "Min average window.")
add_arg('max_average_window',int,   1,     "Max average window. It is proposed to be set as the number of minibatch in a pass.")
add_arg('average_window',    float, 0.15,      "Average window.")
add_arg('parallel',          bool,  False,     "Whether use parallel training.")
add_arg('profile',           bool,  False,      "Whether to use profiling.")
add_arg('skip_batch_num',    int,   0,          "The number of first minibatches to skip as warm-up for better performance test.")
add_arg('skip_test',         bool,  False,      "Whether to skip test phase.")
add_arg('with_accuracy',     bool,  True,      "Whether to save a model with accuracy layers.")
# yapf: enable


def evaluate(args):
    """OCR training"""

    get_feeder_data = get_ctc_feeder_data
    num_classes = data_reader.num_classes()
    data_shape = data_reader.data_shape()

    if args.with_accuracy:
        ids, evaluator = ctc_eval(data_shape, num_classes, args.use_gpu)
    else:
        ids = ctc_eval_noacc(data_shape, num_classes, args.use_gpu)

    # define network

    # prepare environment
    place = fluid.CPUPlace()
    if args.use_gpu:
        place = fluid.CUDAPlace(0)

    if 'ce_mode' in os.environ:
        fluid.default_startup_program().random_seed = 90

    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    # load init model
    if args.init_model is not None:
        model_dir = args.init_model
        model_file_name = None
        if not os.path.isdir(args.init_model):
            model_dir = os.path.dirname(args.init_model)
            model_file_name = os.path.basename(args.init_model)
        fluid.io.load_params(exe, dirname=model_dir, filename=model_file_name)
        print("Init model from: %s." % args.init_model)

    train_exe = exe

    fetch_vars = [ids]
    if args.with_accuracy:
        evaluator.reset(exe)
        fetch_vars += evaluator.metrics
        fetch_vars.append(evaluator.seq_num)
        fetch_vars.append(evaluator.instance_error)

    feeds = ["pixel"]
    if args.with_accuracy:
        feeds.append("label")

    inference_program = fluid.default_main_program().clone(for_test=True)

    fluid.io.save_inference_model(
        args.save_model_dir,
        feeds,
        fetch_vars,
        exe,
        main_program=inference_program)
    print("Model saved!")
    exit()


def main():
    args = parser.parse_args()
    print_arguments(args)
    evaluate(args)


if __name__ == "__main__":
    main()

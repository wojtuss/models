import argparse
import ast
import numpy as np
from functools import partial
import os
import paddle
import paddle.fluid as fluid

import model
from model import wrap_encoder as encoder
from model import wrap_decoder as decoder
from model import fast_decode as fast_decoder
from config import *
from train import pad_batch_data
import reader
import paddle.fluid.profiler as profiler
import time

def parse_args():
    parser = argparse.ArgumentParser("Training for Transformer.")
    parser.add_argument(
        '--display_output',
        type=ast.literal_eval,
        default=False,
        help="Display translation result or not")
    parser.add_argument(
        '--save_output',
        type=ast.literal_eval,
        default=False,
        help="Save output of inference to file")
    parser.add_argument(
        '--save_path',
        type=str,
        default="predict.txt",
        help="The file name we are going to save the output")
    parser.add_argument(
				'--device',
				type=str,
				default='GPU',
				help="GPU or CPU device")
    parser.add_argument(
        "--skip_pass_num",
        type=int,
        default=0,
        help="Profiling the inference but skip the first few passes")
    parser.add_argument(
        "--profile",
        action='store_true',
        help="If set, do profiling")
    parser.add_argument(
        "--num_profiling_passes",
        type=int,
        default=100,
        help="Number of passes that do profiling")
    parser.add_argument(
        "--src_vocab_fpath",
        type=str,
        required=True,
        help="The path of vocabulary file of source language.")
    parser.add_argument(
        "--trg_vocab_fpath",
        type=str,
        required=True,
        help="The path of vocabulary file of target language.")
    parser.add_argument(
        "--test_file_pattern",
        type=str,
        required=True,
        help="The pattern to match test data files.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=50,
        help="The number of examples in one run for sequence generation.")
    parser.add_argument(
        "--pool_size",
        type=int,
        default=10000,
        help="The buffer size to pool data.")
    parser.add_argument(
        "--special_token",
        type=str,
        default=["<s>", "<e>", "<unk>"],
        nargs=3,
        help="The <bos>, <eos> and <unk> tokens in the dictionary.")
    parser.add_argument(
        "--token_delimiter",
        type=lambda x: str(x.encode().decode("unicode-escape")),
        default=" ",
        help="The delimiter used to split tokens in source or target sentences. "
        "For EN-DE BPE data we provided, use spaces as token delimiter. ")
    parser.add_argument(
        'opts',
        help='See config.py for all options',
        default=None,
        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    # Append args related to dict
    src_dict = reader.DataReader.load_dict(args.src_vocab_fpath)
    trg_dict = reader.DataReader.load_dict(args.trg_vocab_fpath)
    dict_args = [
        "src_vocab_size", str(len(src_dict)), "trg_vocab_size",
        str(len(trg_dict)), "bos_idx", str(src_dict[args.special_token[0]]),
        "eos_idx", str(src_dict[args.special_token[1]]), "unk_idx",
        str(src_dict[args.special_token[2]])
    ]
    merge_cfg_from_list(args.opts + dict_args,
                        [InferTaskConfig, ModelHyperParams])
    return args


def post_process_seq(seq,
                     bos_idx=ModelHyperParams.bos_idx,
                     eos_idx=ModelHyperParams.eos_idx,
                     output_bos=InferTaskConfig.output_bos,
                     output_eos=InferTaskConfig.output_eos):
    """
    Post-process the beam-search decoded sequence. Truncate from the first
    <eos> and remove the <bos> and <eos> tokens currently.
    """
    eos_pos = len(seq) - 1
    for i, idx in enumerate(seq):
        if idx == eos_idx:
            eos_pos = i
            break
    seq = [
        idx for idx in seq[:eos_pos + 1]
        if (output_bos or idx != bos_idx) and (output_eos or idx != eos_idx)
    ]
    return seq


def prepare_batch_input(insts, data_input_names, src_pad_idx, bos_idx, n_head,
                        d_model, place):
    """
    Put all padded data needed by beam search decoder into a dict.
    """
    src_word, src_pos, src_slf_attn_bias, src_max_len = pad_batch_data(
        [inst[0] for inst in insts], src_pad_idx, n_head, is_target=False)
    # start tokens
    trg_word = np.asarray([[bos_idx]] * len(insts), dtype="int64")
    trg_src_attn_bias = np.tile(src_slf_attn_bias[:, :, ::src_max_len, :],
                                [1, 1, 1, 1]).astype("float32")
    trg_word = trg_word.reshape(-1, 1, 1)
    src_word = src_word.reshape(-1, src_max_len, 1)
    src_pos = src_pos.reshape(-1, src_max_len, 1)

    def to_lodtensor(data, place, lod=None):
        data_tensor = fluid.LoDTensor()
        data_tensor.set(data, place)
        if lod is not None:
            data_tensor.set_lod(lod)
        return data_tensor

    # beamsearch_op must use tensors with lod
    init_score = to_lodtensor(
        np.zeros_like(
            trg_word, dtype="float32").reshape(-1, 1),
        place, [range(trg_word.shape[0] + 1)] * 2)
    trg_word = to_lodtensor(trg_word, place, [range(trg_word.shape[0] + 1)] * 2)

    data_input_dict = dict(
        zip(data_input_names, [
            src_word, src_pos, src_slf_attn_bias, trg_word, init_score,
            trg_src_attn_bias
        ]))

    input_dict = dict(data_input_dict.items())
    return input_dict


def fast_infer(test_data, trg_idx2word):
    """
    Inference by beam search decoder based solely on Fluid operators.
    """
    place = fluid.CUDAPlace(0) if InferTaskConfig.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)

    out_ids, out_scores = fast_decoder(
        ModelHyperParams.src_vocab_size, ModelHyperParams.trg_vocab_size,
        ModelHyperParams.max_length + 1, ModelHyperParams.n_layer,
        ModelHyperParams.n_head, ModelHyperParams.d_key,
        ModelHyperParams.d_value, ModelHyperParams.d_model,
        ModelHyperParams.d_inner_hid, ModelHyperParams.prepostprocess_dropout,
        ModelHyperParams.attention_dropout, ModelHyperParams.relu_dropout,
        ModelHyperParams.preprocess_cmd, ModelHyperParams.postprocess_cmd,
        ModelHyperParams.weight_sharing, InferTaskConfig.beam_size,
        InferTaskConfig.max_out_len, ModelHyperParams.eos_idx)

    fluid.io.load_vars(
        exe,
        InferTaskConfig.model_path,
        vars=[
            var for var in fluid.default_main_program().list_vars()
            if isinstance(var, fluid.framework.Parameter)
        ])

    # This is used here to set dropout to the test mode.
    infer_program = fluid.default_main_program().clone(for_test=True)

    pass_id = 0
    total_passes =  args.num_profiling_passes + args.skip_pass_num
    batch_latency = [0.0]*total_passes
    batch_wps = [0.0]*total_passes

    for batch_id, data in enumerate(test_data.batch_generator()):
       
        data_input = prepare_batch_input(
            data, encoder_data_input_fields + fast_decoder_data_input_fields,
            ModelHyperParams.eos_idx, ModelHyperParams.bos_idx,
            ModelHyperParams.n_head, ModelHyperParams.d_model, place)
        
        start = time.time()        

        seq_ids, seq_scores = exe.run(infer_program,
                                      feed=data_input,
                                      fetch_list=[out_ids, out_scores],
                                      return_numpy=False)
          
        batch_latency_cur = time.time() - start
        
        if pass_id == args.skip_pass_num:
            profiler.reset_profiler() 
        
        batch_latency[pass_id] = batch_latency_cur
        batch_wps[pass_id] =  args.batch_size / batch_latency_cur  
        if pass_id < args.skip_pass_num:
 						strprefix = "Warm-up pass"
        else:
            strprefix ="Profiling pass"
        print("\n++++  %s, Batch: %d, batch_size: %d, latency: %.5f s, sentence per sec: %f +++" % (strprefix, pass_id, args.batch_size, batch_latency[pass_id], batch_wps[pass_id])) 
        pass_id = pass_id + 1

        # How to parse the results:
        #   Suppose the lod of seq_ids is:
        #     [[0, 3, 6], [0, 12, 24, 40, 54, 67, 82]]
        #   then from lod[0]:
        #     there are 2 source sentences, beam width is 3.
        #   from lod[1]:
        #     the first source sentence has 3 hyps; the lengths are 12, 12, 16
        #     the second source sentence has 3 hyps; the lengths are 14, 13, 15
        hyps = [[] for i in range(len(data))]
        scores = [[] for i in range(len(data))]
        for i in range(len(seq_ids.lod()[0]) - 1):  # for each source sentence
            start = seq_ids.lod()[0][i]
            end = seq_ids.lod()[0][i + 1]
            for j in range(end - start):  # for each candidate
                sub_start = seq_ids.lod()[1][start + j]
                sub_end = seq_ids.lod()[1][start + j + 1]
                hyps[i].append(" ".join([
                    trg_idx2word[idx]
                    for idx in post_process_seq(
                        np.array(seq_ids)[sub_start:sub_end])
                ]))
                scores[i].append(np.array(seq_scores)[sub_end - 1])
                if args.display_output==True:
                    print(hyps[i][-1])
                if args.save_output:
                    with open(args.save_path, 'a') as file:
                        file.write(hyps[i][-1]+'\n')
                if len(hyps[i]) >= InferTaskConfig.n_best:
                    break
        
        if pass_id == total_passes:
            latencies = batch_latency[args.skip_pass_num:]
            latency_avg = np.average(latencies)
            latency_std = np.std(latencies)
            latency_pc99 = np.percentile(latencies, 99)
            wpses = batch_wps[args.skip_pass_num:]
            wps_avg = np.average(wpses)
            wps_std = np.std(wpses)
            wps_pc01 = np.percentile(wpses, 1)
            
            print('\n\nTotal passes (incl. warm-up): %d\n' % (total_passes))
            print('Total examples (incl. warm-up): %d\n' % (total_passes * args.batch_size))
            print('avg latency: %.5f, std latency: %.5f, 99pc latency: %.5f\n' %(latency_avg, latency_std, latency_pc99))
            print('avg wps: %.5f, std wps: %.5f, wps for 99pc latency: %.5f\n' %(wps_avg, wps_std, wps_pc01))

            break

def print_arguments(args):
    print('-----------  Configuration Arguments -----------')
    for arg, value in sorted(vars(args).iteritems()):
        print('%s: %s' % (arg, value))
    print('------------------------------------------------')


def infer(args, inferencer=fast_infer):
    place = fluid.CUDAPlace(0) if InferTaskConfig.use_gpu else fluid.CPUPlace()
    test_data = reader.DataReader(
        src_vocab_fpath=args.src_vocab_fpath,
        trg_vocab_fpath=args.trg_vocab_fpath,
        fpattern=args.test_file_pattern,
        token_delimiter=args.token_delimiter,
        use_token_batch=False,
        batch_size=args.batch_size,
        pool_size=args.pool_size,
        sort_type=reader.SortType.NONE,
        shuffle=False,
        shuffle_batch=False,
        start_mark=args.special_token[0],
        end_mark=args.special_token[1],
        unk_mark=args.special_token[2],
        # count start and end tokens out
        max_length=ModelHyperParams.max_length - 2,
        clip_last_batch=False)
    trg_idx2word = test_data.load_dict(
        dict_path=args.trg_vocab_fpath, reverse=True)
    inferencer(test_data, trg_idx2word)


if __name__ == "__main__":
    args = parse_args()
    print_arguments(args)
    if os.path.exists(args.save_path):
        os.remove(args.save_path)
    if args.profile:
        if args.device == 'GPU':
            with profiler.cuda_profiler("cuda_profiler.txt",'csv') as nvprof:
                infer(args)
        else:
            with profiler.profiler('CPU',sorted_key='total') as cpuprof:
                infer(args)
    else:
        infer(args)

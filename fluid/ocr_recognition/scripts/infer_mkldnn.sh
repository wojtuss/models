#!/bin/bash
FLAGS_use_mkldnn=1 OMP_NUM_THREADS=56 KMP_AFFINITY=granularity=fine,compact,1,0 python ../infer.py \
    --model_path mkldnn_model/model_00001 \
    --input_images_list ~/.cache/paddle/dataset/ctc_data/data/test.list \
    --input_images_dir ~/.cache/paddle/dataset/ctc_data/data/test_images \
    --use_gpu False \
    --batch_size 32 \
    --iterations 5 \
    --skip_batch_num 2
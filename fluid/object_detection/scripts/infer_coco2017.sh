#!/bin/bash
cd ..
time python eval.py \
	--use_gpu False \
	--dataset coco2017 \
	--model_dir model_coco2017/0 \
	--skip_batch_num 2 \
	--iterations 0 \
	--profile
cd -

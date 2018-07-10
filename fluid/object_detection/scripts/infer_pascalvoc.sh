#!/bin/bash
cd ..
time python eval.py \
	--use_gpu False \
	--dataset pascalvoc \
	--model_dir model_pascalvoc/0 \
	--skip_batch_num 2 \
	--iterations 0 \
	--profile
cd -

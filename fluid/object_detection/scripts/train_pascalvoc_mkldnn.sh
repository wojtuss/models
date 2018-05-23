#!/bin/bash
cd ..
time python train.py \
	--use_gpu False \
	--dataset pascalvoc \
	--model_save_dir model_pascalvoc_mkldnn \
	--apply_distort True \
	--apply_expand False \
	--num_passes 1 \
	--iterations 5 \
	--parallel False \
	--use_mkldnn
cd -

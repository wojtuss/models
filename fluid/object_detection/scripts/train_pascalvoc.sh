#!/bin/bash
cd ..
time python train.py \
	--use_gpu False \
	--dataset pascalvoc \
	--model_save_dir model_pascalvoc \
	--num_passes 1 \
	--iterations 5
cd -

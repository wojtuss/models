#!/bin/bash
cd ..
time python train.py \
	--use_gpu False \
	--dataset coco2017 \
	--model_save_dir model_coco2017 \
	--apply_distort True \
	--apply_expand False \
	--num_passes 1 \
	--iterations 5
cd -

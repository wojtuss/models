#!/bin/bash
cd ../
time python train.py \
	--use_gpu False \
	--dataset coco2014 \
	--model_save_dir model_coco2014 \
	--apply_distort True \
	--apply_expand False \
	--num_passes 1 \
	--iterations 5
cd -

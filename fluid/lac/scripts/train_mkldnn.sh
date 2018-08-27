#!/bin/bash
cd ..
export FLAGS_use_mkldnn=True
time python train.py \
	--use_gpu 0 \
	--model_save_dir saved_models \
	--num_iterations 5 \
	--save_model_per_batchs 5
cd -


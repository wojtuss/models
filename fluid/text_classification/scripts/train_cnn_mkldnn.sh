#!/bin/bash
time python ../train.py \
	--device CPU \
	--model_save_dir cnn_model_mkldnn \
	--num_passes 1 \
	--use_mkldnn \
	cnn


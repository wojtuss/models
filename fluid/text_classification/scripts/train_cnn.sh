#!/bin/bash
time python ../train.py \
	--device CPU \
	--model_save_dir cnn_model \
	--num_passes 1 \
	cnn


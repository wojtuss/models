#!/bin/bash
time python ../train.py \
	--device CPU \
	--model_save_dir lstm_model \
	--num_passes 1 \
	lstm


#!/bin/bash
time python ../train.py \
	--device GPU \
	--model_save_dir gpu_model \
	--test_data_dir ../data/test_files \
	--train_data_dir ../data/train_files \
  --parallel True \
	--num_passes 1

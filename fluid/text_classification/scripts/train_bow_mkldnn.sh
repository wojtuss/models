#!/bin/bash
time python ../train.py \
	--device CPU \
	--model_save_dir bow_model_mkldnn \
	--num_passes 1 \
	--use_mkldnn \
	bow


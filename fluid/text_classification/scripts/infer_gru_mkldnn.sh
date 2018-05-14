#!/bin/bash
time python ../infer.py \
	--device CPU \
	--model_path gru_model_mkldnn/epoch0 \
	--num_passes 100 \
	--profile


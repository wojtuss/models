#!/bin/bash
time python ../infer.py \
	--device CPU \
	--model_path bilstm_model/epoch0 \
	--num_passes 100 \
	--profile


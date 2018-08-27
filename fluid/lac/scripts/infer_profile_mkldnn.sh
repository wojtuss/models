#!/bin/bash
cd ..
export FLAGS_use_mkldnn=True
time python infer.py \
	--device CPU \
	--model_path saved_models/params_batch_1 \
	--skip_pass_num 0 \
	--num_passes 100 \
	--profile
cd -

#!/bin/bash
# infer dataset flowers using model resnet_imagenet
time python ../infer_resnet.py \
	--device CPU \
	--batch_size 32 \
	--skip_batch_num 2 \
	--data_set flowers \
	--iterations 100 \
	--infer_model_path ../models/resnet_imagenet_flowers \
	--profile

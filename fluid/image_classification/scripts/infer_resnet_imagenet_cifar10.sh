#!/bin/bash
# infer dataset cifar10 using model resnet_imagenet
time python ../infer_resnet.py \
	--device CPU \
	--batch_size 32 \
	--skip_batch_num 2 \
	--data_set cifar10 \
	--iterations 100 \
	--infer_model_path ../models/resnet_imagenet_cifar10 \
	--profile

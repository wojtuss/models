#!/bin/bash
# infer dataset imagenet using model resnet_cifar10
time python ../infer_resnet.py \
	--device CPU \
	--batch_size 32 \
	--skip_batch_num 2 \
	--data_set imagenet \
	--iterations 100 \
	--infer_model_path ../models/resnet_cifar10_imagenet \
	--profile

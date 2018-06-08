#!/bin/bash
# infer dataset imagenet using model resnet_imagenet
time python ../infer_resnet.py \
	--device CPU \
	--batch_size 32 \
	--skip_batch_num 2 \
	--data_set imagenet \
	--test_file_list ../data/imagenet/val_list.txt \
	--data_dir ../data/imagenet \
	--iterations 100 \
	--infer_model_path ../models/resnet_imagenet_imagenet \
	--profile

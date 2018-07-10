#!/bin/bash
# train model resnet_imagenet using dataset imagenet
time python ../train_resnet.py \
	--device CPU \
	--batch_size 32 \
	--skip_batch_num 0 \
	--pass_num 1 \
	--iterations 1 \
	--model resnet_imagenet \
	--data_set imagenet \
	--train_file_list ../data/imagenet/train_list.txt \
	--test_file_list ../data/imagenet/val_list.txt \
	--data_dir ../data/imagenet \
	--skip_test \
	--save_model \
	--save_model_path ../models/resnet_imagenet_imagenet

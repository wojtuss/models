#!/bin/bash
# train model resnet_imagenet using dataset imagenet
export OMP_NUM_THREADS=2                        
time python ../train_resnet.py \
	--device CPU \
	--batch_size 2 \
	--skip_batch_num 0 \
	--pass_num 1 \
	--iterations 5 \
	--model resnet50 \
	--data_set imagenet \
	--train_file_list /home/wojtuss/repos/PaddlePaddle/data/ImageNet/train_list.txt \
	--test_file_list /home/wojtuss/repos/PaddlePaddle/data/ImageNet/val_list.txt \
	--data_dir /home/wojtuss/repos/PaddlePaddle/data/ImageNet/ \
	--skip_test \
	--save_model \
	--save_model_path ../saved_models/resnet50_ImageNet

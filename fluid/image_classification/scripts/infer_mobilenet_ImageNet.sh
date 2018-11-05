#!/bin/bash
# infer dataset imagenet using MobileNet-v1 model
export FLAGS_use_mkldnn=1
export OMP_NUM_THREADS=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
time python ../infer_image_classification.py \
	--device CPU \
	--batch_size 1 \
	--skip_batch_num 10 \
	--data_set imagenet \
	--test_file_list /home/wojtuss/repos/PaddlePaddle/data/ImageNet/val_list.txt \
	--data_dir /home/wojtuss/repos/PaddlePaddle/data/ImageNet/ \
	--iterations 100 \
	--infer_model_path ../saved_models/MobileNet-v1_baidu \
	--use_transpiler True \
	--profile


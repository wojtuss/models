#!/bin/bash

export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
cd build

	# --infer_model=/home/wojtuss/repos/PaddlePaddle/data/Haihao/resnet50 \
# cgdb --args ./infer_image_classification \
./infer_image_classification \
	--infer_model=/home/wojtuss/repos/PaddlePaddle/data/saved_models/ResNet50/ \
        --data_list=/home/wojtuss/repos/PaddlePaddle/data/ImageNet/val_list.txt \
        --data_dir=/home/wojtuss/repos/PaddlePaddle/data/ImageNet/ \
        --calibration_data_list=/home/wojtuss/repos/PaddlePaddle/data/ImageNet/val_list.txt \
        --calibration_data_dir=/home/wojtuss/repos/PaddlePaddle/data/ImageNet/ \
        --batch_size=50 \
        --calibration_batch_size=50 \
	--paddle_num_threads=14 \
        --skip_batch_num=0 \
        --iterations=10  \
        --profile \
        --use_mkldnn=1 \
	--with_labels=1 \
	--one_file_params=0 \
	--enable_graphviz=1 \
	--use_int8=1 \
	# --skip_passes \
	# --use_fake_data \

cd -


#!/bin/bash
cd ..
python save_model.py \
	--dataset=pascalvoc \
	--model_dir=/home/wojtuss/repos/PaddlePaddle/data/MobileNet-SSD/ssd_mobilenet_v1_pascalvoc \
	--data_dir=/home/wojtuss/repos/PaddlePaddle/data/pascalvoc/ \
	--test_list=/home/wojtuss/repos/PaddlePaddle/data/pascalvoc/test.txt \
	--batch_size=1 \
	--use_gpu=False \
	--model_save_dir=my_super_model

cd -

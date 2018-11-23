#!/bin/bash
export OMP_NUM_THREADS=14
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

cd ..
python eval.py \
	--dataset=pascalvoc \
	--model_dir=/home/wojtuss/repos/PaddlePaddle/data/MobileNet-SSD/ssd_mobilenet_v1_pascalvoc \
	--data_dir=/home/wojtuss/repos/PaddlePaddle/data/pascalvoc/ \
	--test_list=/home/wojtuss/repos/PaddlePaddle/data/pascalvoc/test.txt \
	--batch_size=1 \
	--use_gpu=False \

cd -

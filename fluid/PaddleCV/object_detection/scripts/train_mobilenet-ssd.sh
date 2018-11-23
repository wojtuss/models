#!/bin/bash
export OMP_NUM_THREADS=14
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

cd ..
python train_save_model.py \
	--dataset=pascalvoc \
	--data_dir=/home/wojtuss/repos/PaddlePaddle/data/pascalvoc/ \
	--batch_size=1 \
	--epoc_num=1 \
	--parallel=False \
	--use_gpu=False \
	--enable_ce=True \
	--model_save_dir ./my_model

cd -

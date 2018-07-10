#!/bin/bash
cd ..
time python eval.py \
	--use_gpu False \
	--dataset coco2014 \
	--model_dir model_coco2014_mkldnn/0 \
	--skip_batch_num 2 \
	--iterations 0 \
	--profile \
	--use_mkldnn
cd -

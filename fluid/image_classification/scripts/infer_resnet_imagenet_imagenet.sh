#!/bin/bash
# infer dataset imagenet using model resnet_imagenet
time python ../infer_resnet.py \
	--device CPU \
	--batch_size 32 \
	--skip_batch_num 2 \
	--data_set imagenet \
	--test_file_list /home/kbinias/data/imagenet/val_list.txt \
	--data_dir /home/kbinias/data/imagenet \
	--iterations 50 \
	--infer_model_path /home/lidanqin/models/resnet50_baidu \
	--use_transpiler True \
        #--profile 

#!/bin/bash
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
cd build
./infer_object_detection \
	--infer_model=/home/wojtuss/repos/PaddlePaddle/data/MobileNet-SSD/MobileNet-SSD_pascalvoc_no_acc/ \
        --data_list=/home/wojtuss/repos/PaddlePaddle/data/pascalvoc/test.txt \
        --label_list=/home/wojtuss/repos/PaddlePaddle/data/pascalvoc/label_list \
        --data_dir=/home/wojtuss/repos/PaddlePaddle/data/pascalvoc/ \
        --batch_size=1 \
	--paddle_num_threads=1 \
        --skip_batch_num=0 \
        --iterations=1000  \
	--profile \
        --use_mkldnn=1 \
	--with_labels=0 \
	--enable_graphviz=1

cd -


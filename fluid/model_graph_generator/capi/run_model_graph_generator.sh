#!/bin/bash
cd build
./model_graph_generator \
	--model=/home/wojtuss/repos/PaddlePaddle/data/MobileNet-v1/MobileNet-v1_baidu \
	--one_file_params=0 \
        --use_mkldnn=1 \
	# --skip_passes \

cd -


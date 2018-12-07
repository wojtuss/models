#!/bin/bash
export LD_LIBRARY_PATH=~/repos/PaddlePaddle/Paddle/build/third_party/install/warpctc/lib/
cd ..
python save_model.py \
	--save_model_dir=model_without_acc \
	--with_accuracy=False

cd -


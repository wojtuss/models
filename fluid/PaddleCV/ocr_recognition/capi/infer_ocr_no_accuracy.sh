
#include "paddle/fluid/inference/api/helper.h"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
cd build
./infer_ocr_detection \
  --use_mkldnn=false \
  --data_list=/home/li \
  --image_dir=/home/wojtuss/repos/PaddlePaddle/data/pascalvoc/ \
  --iterations=1000  \
	--infer_model=/home/li/models \
  --batch_size=1 \
  --skip_batch_num=0 \
	--profile \

cd -

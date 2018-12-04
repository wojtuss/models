
#include "paddle/fluid/inference/api/helper.h"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
cd build
./infer_ocr_detection \
  --skip_passes=true \
  --paddle_num_threads=14 \
  --use_mkldnn=true \
  --data_list=/home/li/.cache/paddle/dataset/ctc_data/data/test.list \
  --data_dir=/home/li/.cache/paddle/dataset/ctc_data/data/test_images\
  --iterations=100  \
	--infer_model=/home/li/AIPG-models/paddle-models/fluid/PaddleCV/ocr_recognition/saved_model_no_acc \
  --batch_size=1 \
  --skip_batch_num=0 \
	--profile=true 

cd -

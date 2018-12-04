
#include "paddle/fluid/inference/api/helper.h"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
cd build
./infer_ocr_detection \
  --paddle_num_threads=14 \
  --mkldnn_used=true \
  --data_list=/home/sfraczek/.cache/paddle/dataset/ctc_data/data/test.list \
  --image_dir=/home/sfraczek/.cache/paddle/dataset/ctc_data/data/test_images\
  --iterations=100  \
	--infer_model=/home/sfraczek/source/Paddle-models/fluid/PaddleCV/ocr_recognition/saved_models_no_acc \
  --batch_size=50 \
  --skip_batches=0 \
	--profile=true 

cd -

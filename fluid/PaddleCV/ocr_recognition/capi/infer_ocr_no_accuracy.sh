
#include "paddle/fluid/inference/api/helper.h"
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
cd build
./infer_ocr_detection \
  --mkldnn_used=false \
  --data_list=/home/li/.cache/paddle/dataset/ctc_data/data/test.list \
  --image_dir=/home/li/.cache/paddle/dataset/ctc_data/data/test_images\
  --iterations=10  \
	--infer_model=/home/li/AIPG-models/paddle-models/fluid/PaddleCV/ocr_recognition/saved_models_no_acc \
  --batch_size=1 \
  --skip_batches=0 \
	--profile=true\

cd -

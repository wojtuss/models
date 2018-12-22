export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=1

cd build
./infer_ocr_recognition \
  --infer_model=/home/wojtuss/repos/PaddlePaddle/data/CRNN-CTC/CRNN-CTC_model \
  --data_list=/home/wojtuss/repos/PaddlePaddle/data/CRNN-CTC/data/test.list \
  --data_dir=/home/wojtuss/repos/PaddlePaddle/data/CRNN-CTC/data/test_images \
  --use_mkldnn=true \
  --batch_size=1 \
  --iterations=100  \
  --skip_batch_num=5 \
  --paddle_num_threads=1 \
  --profile=true \
  --enable_graphviz=1 \
  --with_labels=true \
  # --skip_passes \

cd -

export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
python infer.py \
  --model=crnn_ctc \
  --model_path=/home/li/AIPG-models/paddle-models/fluid/PaddleCV/ocr_recognition/models/model_00001 \
  --save_model_dir=saved_model_no_acc \
  --input_images_list=/home/li/.cache/paddle/dataset/ctc_data/data/test.list \
  --input_images_dir=/home/li/.cache/paddle/dataset/ctc_data/data/test_images \
  --use_gpu=False \



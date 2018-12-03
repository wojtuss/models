export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
python eval.py \
  --model=crnn_ctc \
  --model_path=/home/li/AIPG-models/paddle-models/fluid/PaddleCV/ocr_recognition/models/model_00001 \
  --save_model_dir=my_saved_model \
  --input_image_list=/home/li/.cache/paddle/dataset/ctc_data/data/test.list \
  --input_image_dir=/home/li/.cache/paddle/dataset/ctc_data/data/test_images \
  --use_gpu=False \



run_mobilenet_ssd(){
cgdb --args ./infer_object_detection \
#./infer_object_detection \
  --infer_model=/home/lidanqin/AIPG-models/paddle-models/fluid/PaddleCV/object_detection/pretrained/MobileNet-SSD_pascalvoc \
        --data_list=/home/lidanqin/data/pascalvoc/test.txt \
        --data_dir=/home/lidanqin/data/pascalvoc/ \
        --label_list=/home/lidanqin/data/pascalvoc/label_list \
        --batch_size=2 \
  --paddle_num_threads=1 \
        --skip_batch_num=0 \
        --iterations=4  \
        --profile \
        --use_mkldnn=1 \
  --with_labels=1 \
  --one_file_params=0 \
  --enable_graphviz=1 \
  #--result_file=result.json

  # --infer_model=/home/wojtuss/repos/PaddlePaddle/data/MobileNet-SSD/MobileNet-SSD_pascalvoc/ \
  # --skip_passes
}
export OMP_NUM_THREADS=14
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
cd build
run_mobilenet_ssd
cd -


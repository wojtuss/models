export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=12
cd build
#cgdb --args ./infer_neural_machine_translation \
./infer_neural_machine_translation \
  --infer_model=/home/li/models/iter_100000.infer.model \
  --all_vocab_fpath=/home/li/data/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 \
  --test_file_path=/home/li/data/wmt16_ende_data_bpe_clean/newstest2016.tok.bpe.32000.en-de \
  --batch_size=8 \
  --use_mkldnn=true \
  --skip_passes=false \
  --enable_graphviz=1 \
  --paddle_num_threads=12 \
  --profile=true \
  --with_labels=false \
  --beam_size=4 \
  --max_out_len=255 \
  --iterations=200000  \
  --skip_batch_num=0 \
  --output_file=./output_file.txt \

cd -
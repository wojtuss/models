export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=1

cd build
./infer_neural_machine_translation \
  --infer_model=/home/wojtuss/repos/PaddlePaddle/data/Transformer/Transformer_model \
  --all_vocab_fpath=/home/wojtuss/repos/PaddlePaddle/data/Transformer/gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
  --test_file_pattern=/home/wojtuss/repos/PaddlePaddle/data/Transformer/gen_data/wmt16_ende_data_bpe/newstest2016.tok.bpe.32000.en-de \
  --token_delimiter=' ' \
  --batch_size=1 \
  --use_mkldnn=true \
  --skip_passes=true \
  --enable_graphviz=1 \
  --paddle_num_threads=1 \
  --profile=true \
  --with_labels=false \
  --beam_size=4 \
  --max_out_len=255 \
  --iterations=100  \
  --skip_batch_num=5 \
  --output_file=./output_file.txt \

cd -

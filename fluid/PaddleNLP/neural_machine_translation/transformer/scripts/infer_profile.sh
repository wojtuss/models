export FLAGS_use_mkldnn=1
export OMP_NUM_THREADS=1
#export KMP_AFFINITY=granularity=fine,compact,1,0
#export KMP_BLOCKTIME=1   

#gdb --args python -u ../infer.py \
  python ../infer_profile.py \
  --display_output True \
  --save_output True \
  --device CPU \
  --skip_pass_num 3 \
  --profile \
  --num_profiling_passes 5 \
  --src_vocab_fpath ~/data/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 \
  --trg_vocab_fpath ~/data/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 \
  --special_token '<s>' '<e>' '<unk>' \
  --test_file_pattern ~/data/wmt16_ende_data_bpe_clean/newstest2016.tok.bpe.32000.en-de \
  --token_delimiter ' ' \
  --batch_size 8 \
  model_path ~/models/iter_100000.infer.model \
  beam_size 4 \
  max_out_len 255 \
  use_gpu False


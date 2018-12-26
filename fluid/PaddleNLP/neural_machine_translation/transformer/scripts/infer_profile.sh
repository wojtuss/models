export FLAGS_use_mkldnn=0
export OMP_NUM_THREADS=1
#export KMP_AFFINITY=granularity=fine,compact,1,0
#export KMP_BLOCKTIME=1   

#python ../infer_profile.py \
python -m pdb ../infer_profile.py \
  --save_output True \
  --display_output True \
  --device CPU \
  --skip_pass_num 5 \
  --profile \
  --num_profiling_passes 120 \
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


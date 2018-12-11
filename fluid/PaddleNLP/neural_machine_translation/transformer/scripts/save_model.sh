#!/bin/bash
cd ..
python infer.py \
        --src_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
        --trg_vocab_fpath gen_data/wmt16_ende_data_bpe/vocab_all.bpe.32000 \
	--special_token '<s>' '<e>' '<unk>' \
	--test_file_pattern gen_data/wmt16_ende_data_bpe/newstest2016.tok.bpe.32000.en-de \
	--token_delimiter ' ' \
        --batch_size 32 \
	--save_model_dir /home/wojtuss/repos/PaddlePaddle/data/Transformer/my_model \
	model_path /home/wojtuss/repos/PaddlePaddle/data/Transformer/iter_100000.infer.model \
	beam_size 4 \
	max_out_len 255

cd -

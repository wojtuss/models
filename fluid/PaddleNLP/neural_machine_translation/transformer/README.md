
# To profile python version transformer 

## 1. Download the data, models and mosesdecoder, run:
```
bash ../scripts/download_data_model.sh
```
copy the \_\_model\_\_ file at [ PaddleNLP/neural_machine_translation/transformer/saved_model] to iter_100000.infer.model/

model folder is iter_100000.infer.model/

data foler for inference is wmt16_ende_data_clean

mosesdecoder folder is mosesdecoder

Optional: model weights and data above is recently provided by Baidu, there are also formal dataset one could download via transformer/gen_data.sh, which include training dataset, but it is not needed in capi and python inference

## 2. Run
If everything built successfully, you can inference.

An exemplary command is scripts/infer_profile.sh
```
export FLAGS_use_mkldnn=0
export OMP_NUM_THREADS=1
#export KMP_AFFINITY=granularity=fine,compact,1,0
#export KMP_BLOCKTIME=1

#python -m pdb ../infer_profile.py \
python ../infer_profile.py \
  --save_output True \
  --display_output False \
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
```
To add profiling, use the `--profile` option.

To display output on stdout, use option `--display_output True`

To run inference without running passes, use option `--skip_pass_num`.

## 3. Accuracy(BLEU) measurement
Baidu provide BLEU = 33.06 as the translation score reference, refer to [https://github.intel.com/AIPG/paddle-models/blob/develop/fluid/PaddleNLP/neural_machine_translation/transformer/README_cn.md]

We get predict.txt in folder scripts/, and then We generate BLEU score by
```
sed -r 's/(@@ )|(@@ ?$)//g' predict.txt > predict.tok.txt
perl /home/li/data/gen_data/mosesdecoder/scripts/generic/multi-bleu.perl  /home/li/data/gen_data/wmt16_ende_data/newstest2016.tok.de < predict.tok.txt
```
---

# Attention is All You Need: A Paddle Fluid implementation

This is a Paddle Fluid implementation of the Transformer model in [Attention is All You Need]() (Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin, arxiv, 2017).

If you use the dataset/code in your research, please cite the paper:

```text
@inproceedings{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  booktitle={Advances in Neural Information Processing Systems},
  pages={6000--6010},
  year={2017}
}
```

### TODO

This project is still under active development.

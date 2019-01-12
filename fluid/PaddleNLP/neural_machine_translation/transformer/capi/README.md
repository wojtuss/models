# C-API inference application
The `infer_neural_machine_translation` application serves for running C-API based
inference benchmark for transformer model.

# How to build C-API application
In order to build C-API inference application follow these three steps:
1. build paddle.
2. build paddle's target `inference_lib_dist`.
3. build capi inference application.
4. prepare models for inference.

Each one will be shortly described below.
## 1. Build paddle
Do it as you usually do it. In case you never did it, here are example instructions:
```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle 
Newest public paddle sometimes have problem with mkldnn, if paddle reports error, reset HEAD to commit 550e7e410b21484552d30caeedca965ff3a540b0
mkdir build
cd build
cmake .. -DWITH_DOC=OFF -DWITH_GPU=OFF -DWITH_DISTRIBUTE=OFF -DWITH_MKLDNN=ON -DWITH_GOLANG=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DWITH_PROFILER=ON -DWITH_FLUID_ONLY=ON -DON_INFER=ON
make -j <num_cpu_cores>
```
## 2. Build paddle's target `inference_lib_dist`
While still staying in `/path/to/Paddle/build`, build the target `inference_lib_dist`:
```
make -j <num_cpu_cores> inference_lib_dist
```
Now a directory named `fluid_install_dir` should exist in the build directory.
Remember that path.

## 3. Build C-API inference application
Go to the location of this README.md (PaddleNLP/neural_machine_translation/transformer/capi)
```
mkdir build
cd build
cmake .. -DPADDLE_ROOT=/path/to/Paddle/build/fluid_install_dir
make
```

## 4. Prepare data and models
Download the data, models and mosesdecoder, run:
```
bash ../scripts/download_data_model.sh  
```
copy the \_\_model\_\_ file at [ PaddleNLP/neural_machine_translation/transformer/saved_model] to iter_100000.infer.model/ 

model folder is iter_100000.infer.model/   

data foler for inference is wmt16_ende_data_clean  

mosesdecoder folder is mosesdecoder  

Optional: model weights and data above is recently provided by Baidu, there are also formal dataset one could download via transformer/gen_data.sh, which include training dataset, but it is not needed in capi and python inference

## 5. Run
If everything built successfully, you can inference.

An exemplary command: capi/run_inference_transformer.sh
```
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1
export OMP_NUM_THREADS=12
cd build
./infer_neural_machine_translation \
  --infer_model=/home/li/models/iter_100000.infer.model \
  --all_vocab_fpath=/home/li/data/wmt16_ende_data_bpe_clean/vocab_all.bpe.32000 \
  --test_file_path=/home/li/data/wmt16_ende_data_bpe_clean/newstest2016.tok.bpe.32000.en-de \
  --batch_size=1 \
  --use_mkldnn=true \
  --skip_passes=false \
  --enable_graphviz=1 \
  --paddle_num_threads=12 \
  --profile=true \
  --with_labels=false \
  --beam_size=4 \
  --max_out_len=255 \
  --iterations=3000  \
  --skip_batch_num=0 \
  --output_file=./output_file.txt \

sed -r 's/(@@ )|(@@ ?$)//g' output_file.txt > output.tok.txt
perl /home/li/data/mosesdecoder/scripts/generic/multi-bleu.perl  /home/li/data/wmt16_ende_data_bpe_clean/newstest2016.tok.de < output.tok.txt
cd -
```
To add profiling, use the `--profile` option.

To run inference without running passes, use option `--skip_passes`.

To switch on or off mkldnn, use option `--use_mkldnn` 

## 6. Accuracy(BLEU) measurement
Baidu provide BLEU = 33.64 for newstest2016, source: [https://github.intel.com/AIPG/paddle-models/blob/develop/fluid/PaddleNLP/neural_machine_translation/transformer/README_cn.md]

I tested 3000 iterations(sentences), achieving accuracy:
BLEU = 33.64, 64.6/39.7/27.1/19.0 (BP=0.992, ratio=0.992, hyp_len=61892, ref_len=62362)



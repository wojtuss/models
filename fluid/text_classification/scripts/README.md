## Purpose of this directory
The purpose of this directory is to provide exemplary execution commands. They are inside bash scripts described below.

## Preparation
To add execution permissions for shell scripts, run in this directory:
`chmod +x *.sh`

## Performance tips
Use the below environment flags for best performance:
```
KMP_AFFINITY=granularity=fine,compact,1,0
OMP_NUM_THREADS=<num_of_physical_cores>
```
For example, you can export them, or add them inside the specific files.

## Training
The training is run for 1 pass measuring time by `time` program.
### CPU with mkldnn
Depending on the model you want to profile, run on of:
```
train_bow_mkldnn.sh
train_cnn_mkldnn.sh
train_gru_mkldnn.sh
train_lstm_mkldnn.sh
```
### CPU without mkldnn
Depending on the model yuu want to profile, run one of:
```
train_bow.sh
train_cnn.sh
train_gru.sh
train_lstm.sh
```

## Inference
The inference is run for 100 passes with profiling and measuring time by `time` program.
### CPU with mkldnn
Depending on the model yuu want to profile, run one of:
```
infer_bow_mkldnn.sh
infer_cnn_mkldnn.sh
infer_gru_mkldnn.sh
infer_lstm_mkldnn.sh
```
### CPU without mkldnn
Depending on the model yuu want to profile, run one of:
```
infer_bow.sh
infer_cnn.sh
infer_gru.sh
infer_lstm.sh
```
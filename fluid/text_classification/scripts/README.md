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
FLAGS_use_mkldnn=True
```
For example, you can export them, or add them inside the specific files.
Using FLAGS_use_mkldnn=True is required for launching on mkldnn.

## Training
The training is run for 1 pass measuring time by `time` program.
### CPU 
Depending on the model you want to profile, run one of:
```
train_bow.sh
train_cnn.sh
train_gru.sh
train_lstm.sh
```

## Inference
The inference is run for 100 passes with profiling and measuring time by `time` program.
### CPU 
Depending on the model you want to profile, run one of:
```
infer_bow.sh
infer_cnn.sh
infer_gru.sh
infer_lstm.sh
```

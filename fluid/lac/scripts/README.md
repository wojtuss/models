## Purpose of this directory
The purpose of this directory is to provide exemplary execution commands. The commands are inside bash scripts described below.

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
### CPU with mkldnn
Run:
`train_mkldnn.sh`
### CPU without mkldnn
Run:
`train.sh`

## Inference
### CPU with mkldnn, with profiling
Run:
`infer_profile_mkldnn.sh`
### CPU without mkldnn, with profiling
Run:
`infer_profile.sh`

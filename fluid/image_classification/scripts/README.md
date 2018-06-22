## Purpose of this directory
The purpose of this directory is to provide exemplary execution commands. They are inside bash scripts described below.

## Preparation
To add execution permissions for shell scripts, run in this directory:
```
chmod +x *.sh
```

To be able to run **any script**, add 
```
--use_fake_data
```
argument inside it!

## Performance tips
Use the below environment flags for best performance:
```
KMP_AFFINITY=granularity=fine,compact,1,0
OMP_NUM_THREADS=<num_of_physical_cores>
```
For example, you can export them, or add them inside the specific files.

## Training
The training is run for 1 pass of 1 or 100 iterations (depending on the chosen script) measuring time by `time` program.
The scripts are named by appending model and dataset names.  
### CPU without mkldnn
Depending on the model and dataset you want, run one of:
```
train_resnet_cifar10_cifar10.sh
train_resnet_cifar10_flowers.sh
train_resnet_cifar10_imagenet.sh
train_resnet_imagenet_cifar10.sh
train_resnet_imagenet_flowers.sh
train_resnet_imagenet_imagenet.sh
```

## Inference
The inference is run for 1 pass of 100 iterations measuring time by `time` program.
The scripts are named by appending model and dataset names.  
### CPU without mkldnn
Depending on the model and dataset you want, run one of:
```
infer_resnet_cifar10_cifar10.sh
infer_resnet_cifar10_flowers.sh
infer_resnet_cifar10_imagenet.sh
infer_resnet_imagenet_cifar10.sh
infer_resnet_imagenet_flowers.sh
infer_resnet_imagenet_imagenet.sh
```

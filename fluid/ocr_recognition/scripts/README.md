## Introduction
Scripts enclosed in the folder serve as examples of commands that start training 
and inference of a model, and are subject to further customisation.

## Prerequisites
In order to run the training and inference, no special requirements are posed.

## Training
To run training on *CPU* please execute:

```sh
source train_cpu.sh
```

To run training on *GPU* please execute:

```sh
source train_gpu.sh
```

## Inference
To perform inference on the trained model using *CPU*, please run:
```sh
source infer_cpu.sh
```

To perform inference on the trained model using *GPU*, please run:

```sh
source infer_gpu.sh
```
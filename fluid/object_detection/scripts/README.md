## Purpose of this directory
The purpose of this directory is to provide exemplary execution commands. They are inside bash scripts described below.

## Preparation
To add execution permissions for shell scripts, run in this directory:
```
chmod +x *.sh
```

To be able to run training on coco, install pycocotools and cython:
```
sudo pip install cython pycocotools
```

Download required dataset and pre-trained model as instructed in README.md in the directory below:
```
# download datasets
cd ../data/pascalvoc
chmod +x download.sh
./download.sh
cd ../coco
chmod +x download.sh
./download.sh
cd ../../
# download pretrained models
chmod +x ./pretrained/download_coco.sh
./pretrained/download_coco.sh
chmod +x ./pretrained/download_imagenet.sh
./pretrained_download_imagenet.sh
# return to this directory
cd scripts
```

## Performance tips
Use the below environment flags for best performance:
```
KMP_AFFINITY=granularity=fine,compact,1,0
OMP_NUM_THREADS=<num_of_physical_cores>
```
For example, you can export them, or add them inside the specific files.

## Training
The training is run for 1 pass of 5 iterations measuring time by `time` program.
### CPU with mkldnn
Depending on the dataset you want to train on, run one of:
```
train_pascalvoc_mkldnn.sh
train_coco2017_mkldnn.sh
train_coco2014_mkldnn.sh
```
### CPU without mkldnn
Depending on the dataset you want to train on, run one of:
```
train_pascalvoc.sh
train_coco2017.sh
train_coco2014.sh
```

## Inference
The training is run for 1 pass measuring time by `time` program.
### CPU with mkldnn
Depending on the dataset you want to train on, run one of:
```
infer_pascalvoc_mkldnn.sh
infer_coco2017_mkldnn.sh
infer_coco2014_mkldnn.sh
```
### CPU without mkldnn
Depending on the dataset you want to train on, run one of:
```
infer_pascalvoc.sh
infer_coco2017.sh
infer_coco2014.sh
```

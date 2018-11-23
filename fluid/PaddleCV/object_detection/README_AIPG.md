## SSD Object Detection

# For testing the eval.py, train.py and infer.py
## pascalvoc
### data download
If you want to put the dataset in other place (not in the project model), copy data/* to the dataset destination. For pascalvoc, run download.sh(create_list.py should be under the same folder after copy)
### model
Model provided by baidu is in the pretrained already
### run eval script for pascalvoc
python eval.py --dataset='pascalvoc' --model_dir='pretrained/ssd_mobilenet_v1_pascalvoc' --data_dir='/home/lidanqin/data/pascalvoc' --test_list='/home/lidanqin/data/pascalvoc/test.txt' --ap_version='11point' --nms_threshold=0.45 --batch_size=1 --use_gpu=False
### Proper output
Batch 0, map [1.0000002]   
Batch 1, map [0.9999998]   
Batch 2, map [0.5999999]   
Batch 3, map [0.6228954]   
Batch 4, map [0.7762622]   
Batch 5, map [0.7938758]   
Batch 6, map [0.8167784]   
### Run train script
export FLAGS_paddle_num_threads=1   
export OMP_NUM_THREADS=1   
python train.py --data_dir=/home/lidanqin/data/pascalvoc --batch_size=1 --parallel=False --pretrained_model=pretrained/ssd_mobilenet_v1_pascalvoc --use_gpu=False --enable_ce=True #To evaluate the model
### Proper result
Batch 0, map [1.0000002]   
Batch 10, map [0.78677523]   
Batch 20, map [0.8414355]   
Batch 30, map [0.8092389]   
Batch 40, map [0.8074479]   
### Run infer script
python infer.py --dataset='pascalvoc' --nms_threshold=0.45 --model_dir='pretrained/ssd_mobilenet_v1_pascalvoc' --image_path='/home/lidanqin/data/pascalvoc/VOCdevkit/VOC2007/JPEGImages/009963.jpg' --use_gpu=False

### Proper result
Image with bbox drawed saved as 009963.jpg, it  should appear in the same folder with infer.py file.

## coco
### Lib preparation
- [Git clone cocoapi and make in PythonAPI](https://github.com/cocodataset/cocoapi)   
- pip install numpy==1.14.0 (1.12.0 and 1.13.0 not good)
### data download 
If you want to put the dataset in other place (not in the project folder), copy data/coco/* to the dataset destination, run download.sh. Coco is big, downloading take a few hours.
### model
Model is already in the pretrained folder
### run eval script for coco2014
python eval.py --use_gpu=False --dataset='coco2014' --model_dir='pretrained/ssd_mobilenet_v1_coco' --data_dir='/home/lidanqin/data/coco' --ap_version='11point' --nms_threshold=0.45 --batch_size=1
### Result
Batch 0, map [0.]   
Batch 10, map [0.00100495]   
Batch 20, map [0.00047847]   
Batch 30, map [0.00030404]   
Batch 40, map [0.00020568]   
Batch 50, map [0.00026582]    

## Table of Contents
- [Introduction](#introduction)
- [Data Preparation](#data-preparation)
- [Train](#train)
- [Evaluate](#evaluate)
- [Infer and Visualize](#infer-and-visualize)
- [Released Model](#released-model)


### Introduction

[Single Shot MultiBox Detector (SSD)](https://arxiv.org/abs/1512.02325) framework for object detection can be categorized as a single stage detector. A single stage detector simplifies object detection as a regression problem, which directly predicts the bounding boxes and class probabilities without region proposal. SSD further makes improves by producing these predictions of different scales from different layers, as shown below. Six levels predictions are made in six different scale feature maps. And there are two 3x3 convolutional layers in each feature map, which predict category or a shape offset relative to the prior box(also called anchor), respectively. Thus, we get 38x38x4 + 19x19x6 + 10x10x6 + 5x5x6 + 3x3x4 + 1x1x4 = 8732 detections per class.
<p align="center">
<img src="images/SSD_paper_figure.jpg" height=300 width=900 hspace='10'/> <br />
The Single Shot MultiBox Detector (SSD)
</p>

SSD is readily pluggable into a wide variant standard convolutional network, such as VGG, ResNet, or MobileNet, which is also called base network or backbone. In this tutorial we used [MobileNet](https://arxiv.org/abs/1704.04861).


### Data Preparation

You can use [PASCAL VOC dataset](http://host.robots.ox.ac.uk/pascal/VOC/) or [MS-COCO dataset](http://cocodataset.org/#download).

If you want to train a model on PASCAL VOC dataset, please download dataset at first, skip this step if you already have one.

```bash
cd data/pascalvoc
./download.sh
```

The command `download.sh` also will create training and testing file lists.

If you want to train a model on MS-COCO dataset, please download dataset at first, skip this step if you already have one.

```
cd data/coco
./download.sh
```

### Train

#### Download the Pre-trained Model.

We provide two pre-trained models. The one is MobileNet-v1 SSD trained on COCO dataset, but removed the convolutional predictors for COCO dataset. This model can be used to initialize the models when training other datasets, like PASCAL VOC. The other pre-trained model is MobileNet-v1 trained on ImageNet 2012 dataset but removed the last weights and bias in the Fully-Connected layer.

Declaration: the MobileNet-v1 SSD model is converted by [TensorFlow model](https://github.com/tensorflow/models/blob/f87a58cd96d45de73c9a8330a06b2ab56749a7fa/research/object_detection/g3doc/detection_model_zoo.md). The MobileNet-v1 model is converted from [Caffe](https://github.com/shicai/MobileNet-Caffe).
We will release the pre-trained models by ourself in the upcoming soon.

  - Download MobileNet-v1 SSD:
    ```bash
    ./pretrained/download_coco.sh
    ```
  - Download MobileNet-v1:
    ```bash
    ./pretrained/download_imagenet.sh
    ```

#### Train on PASCAL VOC

`train.py` is the main caller of the training module. Examples of usage are shown below.
  ```bash
  python -u train.py --batch_size=64 --dataset='pascalvoc' --pretrained_model='pretrained/ssd_mobilenet_v1_coco/'
  ```
   - Set ```export CUDA_VISIBLE_DEVICES=0,1``` to specifiy the number of GPU you want to use.
   - Set ```--dataset='coco2014'``` or ```--dataset='coco2017'``` to train model on MS COCO dataset.
   - For more help on arguments:

  ```bash
  python train.py --help
  ```

Data reader is defined in `reader.py`. All images will be resized to 300x300. In training stage, images are randomly distorted, expanded, cropped and flipped:
   - distort: distort brightness, contrast, saturation, and hue.
   - expand: put the original image into a larger expanded image which is initialized using image mean.
   - crop: crop image with respect to different scale, aspect ratio, and overlap.
   - flip: flip horizontally.

We used RMSProp optimizer with mini-batch size 64 to train the MobileNet-SSD. The initial learning rate is 0.001, and was decayed at 40, 60, 80, 100 epochs with multiplier 0.5, 0.25, 0.1, 0.01, respectively. Weight decay is 0.00005. After 120 epochs we achieve 73.32% mAP under 11point metric.

### Evaluate

You can evaluate your trained model in different metrics like 11point, integral on both PASCAL VOC and COCO dataset. Note we set the default test list to the dataset's test/val list, you can use your own test list by setting ```--test_list``` args.

`eval.py` is the main caller of the evaluating module. Examples of usage are shown below.
```bash
python eval.py --dataset='pascalvoc' --model_dir='train_pascal_model/best_model' --data_dir='data/pascalvoc' --test_list='test.txt' --ap_version='11point' --nms_threshold=0.45
```

You can set ```--dataset``` to ```coco2014``` or ```coco2017``` to evaluate COCO dataset. Moreover, we provide `eval_coco_map.py` which uses a COCO-specific mAP metric defined by [COCO committee](http://cocodataset.org/#detections-eval). To use this eval_coco_map.py, [cocoapi](https://github.com/cocodataset/cocoapi) is needed.
Install the cocoapi:
```
# COCOAPI=/path/to/clone/cocoapi
git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
cd $COCOAPI/PythonAPI
# Install into global site-packages
make install
# Alternatively, if you do not have permissions or prefer
# not to install the COCO API into global site-packages
python2 setup.py install --user
```

### Infer and Visualize
`infer.py` is the main caller of the inferring module. Examples of usage are shown below.
```bash
python infer.py --dataset='pascalvoc' --nms_threshold=0.45 --model_dir='train_pascal_model/best_model' --image_path='./data/pascalvoc/VOCdevkit/VOC2007/JPEGImages/009963.jpg'
```
Below are the examples of running the inference and visualizing the model result.
<p align="center">
<img src="images/009943.jpg" height=300 width=400 hspace='10'/>
<img src="images/009956.jpg" height=300 width=400 hspace='10'/>
<img src="images/009960.jpg" height=300 width=400 hspace='10'/>
<img src="images/009962.jpg" height=300 width=400 hspace='10'/> <br />
MobileNet-v1-SSD 300x300 Visualization Examples
</p>


### Released Model


| Model                    | Pre-trained Model  | Training data    | Test data    | mAP |
|:------------------------:|:------------------:|:----------------:|:------------:|:----:|
|[MobileNet-v1-SSD 300x300](http://paddlemodels.bj.bcebos.com/ssd_mobilenet_v1_pascalvoc.tar.gz) | COCO MobileNet SSD | VOC07+12 trainval| VOC07 test   | 73.32%  |

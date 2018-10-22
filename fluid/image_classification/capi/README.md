# Build paddle
Build a specific target:
```
make -j <num_cpu_cores> inference_lib_dist
```
# Build capi inference test application
How to build:
```
mkdir build
cd build
cmake .. -DPADDLE_ROOT=/home/sfraczek/source/Paddle/build.debug/fluid_install_dir
make -j <num_cpu_cores>
```
# Run
You can use a script like below:
```
#!/bin/bash
OMP_NUM_THREADS=14 \
./infer_image_classification \
        --infer_model=/home/sfraczek/source/Paddle-models/fluid/image_classification/output/resnet \
        --batch_size=50 \
        --skip_batch_num=0 \
        --iterations=10  \
        --profile \
        --data_list=/home/sfraczek/source/data/ILSVRC2012/val_list.txt \
        --data_dir=/home/sfraczek/source/data/ILSVRC2012/ \
        --use_MKLDNN
```
The command as above requires the inference model (passed via the `infer_model`
option) to return accuracy as the second output and model parameters to be
stored in separate files.

To run inference on a model without accuracy, with parameters stored
in a single file, and with input image size 318x318, run:
```
#!/bin/bash
OMP_NUM_THREADS=14 \
./infer_image_classification \
        --infer_model=/home/sfraczek/source/Paddle-models/fluid/image_classification/output/resnet \
        --batch_size=50 \
        --skip_batch_num=0 \
        --iterations=10  \
        --profile \
        --data_list=/home/sfraczek/source/data/ILSVRC2012/val_list.txt \
        --data_dir=/home/sfraczek/source/data/ILSVRC2012/ \
        --use_MKLDNN \
	--with_labels=0 \
	--one_file_params=1 \
	--resize_size=318 \
	--crop_size=318
```

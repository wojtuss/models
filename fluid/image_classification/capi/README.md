# Build paddle
Build a specific target:
```
make -j <num_cpu_cores> inference_lib_dist
```
# Build capi inference test application
How to build:
```
cmake .. -DPADDLE_ROOT=/home/sfraczek/source/Paddle/build.debug/fluid_install_dir
make -j <num_cpu_cores>
```
# Run
You can use a script like below:
```
#!/bin/bash
cd build.debug
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
cd -
```

# C-API model graph generator
The `model_graph_generator` application serves for generating .dot files with
model graph tree.

# How to build C-API application
In order to build the application:
1. build paddle.
2. build paddle's target `fluid_lib_dist`.
3. build capi the `model_graph_generator` application.

## 1. Build paddle
Do it as you usually do it. In case you never did it, here are example instructions:
```
git clone https://github.com/PaddlePaddle/Paddle.git
cd Paddle
mkdir build
cd build
cmake .. -DWITH_DOC=OFF -DWITH_GPU=OFF -DWITH_DISTRIBUTE=OFF -DWITH_MKLDNN=ON -DWITH_GOLANG=OFF -DWITH_SWIG_PY=ON -DWITH_TESTING=ON -DCMAKE_BUILD_TYPE=Debug -DWITH_TIMER=OFF -DWITH_PROFILER=OFF -DWITH_FLUID_ONLY=ON -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
make -j <num_cpu_cores>
```
## 2. Build paddle's target `fluid_lib_dist`
While still staying in `/path/to/Paddle/build`, build the target `fluid_lib_dist`:
```
make -j <num_cpu_cores> fluid_lib_dist
```
Now a directory should exist in build directory named `fluid_install_dir`. Remember that path.
## 3. Build C-API model graph generator
Go to where this README.md is and execute:
```
mkdir build
cd build
cmake .. -DPADDLE_ROOT=/path/to/Paddle/build/fluid_install_dir
make
```
# Run
If everything builds successfully, you can run the application:
```
#!/bin/bash
./model_graph_generator \
        --model=<path_to_directory_with_model> \
        --use_mkldnn
```
A bunch of .dot files will be generated, one after each pass.

If you want to skip running all the passes, add the `--skip_passes` option to the command above. A model graph for the original model will be generated only.

To run the application with a model which has parameters stored in a single file, add the `--one_file_params=1` option.

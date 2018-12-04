## Dataset
run python train.py without any arguments, dataset will be downloaded automatically to /home/username/.cache/, for example /home/li/.cache/paddle/dataset/ctc\_data  

## python profile
run infer.sh   
Change the data and model diretory. Data has been downloaded in last step. Model has been uploaded there  

## capi profile  
build paddlepaddle and do cmake as object detection capi. [https://github.intel.com/AIPG/paddle-models/tree/develop-integration/fluid/PaddleCV/object\_detection/capi]    
If you have problem with warpctc library. you may add this to .bashrc: 
export LD\_LIBRARY\_PATH=repolocation/paddlepaddle/build/third\_party/install/warpctc/lib
run infer\_ocr\_no\_accuracy.sh, the model has been uploaded there. change us\_mkldnn=true or false will give different speed.   

We are working on with accuracy version. It should be ready soon.

DATA_ROOT=$HOME/data
MODELS_ROOT=$HOME/models

DATA_PATH=$DATA_ROOT/wmt16_ende_data_bpe_clean
MODEL_PATH=$MODELS_ROOT/iter_100000.infer.model
MOSES_PATH=$DATA_ROOT/mosesdecoder

if [ ! -d $DATA_PATH ] ; then
echo "Data doesn't exist, downloading the dataset"

if [ ! -d $DATA_ROOT ] ; then
echo "DATA_ROOT directoy doesnt exist, now create this directory"
mkdir $DATA_ROOT
fi

cd $DATA_ROOT
wget --output-document=wmt16_ende_data.tar.gz http://transformer-model-data.bj.bcebos.com/wmt16_ende_data_bpe_clean.tar.gz
echo "unzip the dataset"
tar xvf wmt16_ende_data.tar.gz
rm wmt16_ende_data.tar.gz
fi

if [ ! -d $MOSES_PATH ]; then
echo "mosesdecoder doesnt exist, cloning moses for data processing"
git clone https://github.com/moses-smt/mosesdecoder.git ${MOSES_PATH}
fi


if [ ! -d $MODEL_PATH ] ; then
echo "Model doesn't exist, downloading the model"

if [ ! -d $MODELS_ROOT ] ; then
echo "MODELS_ROOT directory doesnt exist, now create this directory"
mkdir $MODELS_ROOT
fi

cd $MODELS_ROOT
wget --output-document=iter_100000_infer_model.tar.gz http://transformer-model-data.bj.bcebos.com/iter_100000.infer.model.tar.gz
echo "unzip the model"
tar xvf iter_100000_infer_model.tar.gz
rm iter_100000_infer_model.tar.gz
fi


echo dataset at $DATA_PATH
echo model at $MODEL_PATH
echo mosesdecoder at $MOSES_PATH
echo All Downloaded!

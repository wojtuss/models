export FLAGS_use_mkldnn=0
#export OMP_NUM_THREADS=1
#export KMP_AFFINITY=granularity=fine,compact,1,0
#export KMP_BLOCKTIME=1

time python ../train_se_resnext.py \
	--model 'SE_ResNeXt50_32x4d'\
	--device 'CPU' \
	--batch_size 32 \
	--iterations 3 \
	--pass_num 1 \
	--save_model \
	--save_model_path '../output/SE_ResNeXt50_32x4d'\
	--train_file_list '/home/kbinias/data/imagenet/val_list.txt' \
	--test_file_list '/home/kbinias/data/imagenet/val_list.txt' \
	--data_dir '/home/kbinias/data/imagenet' \

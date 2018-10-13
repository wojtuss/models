export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

time python ../infer_se_resnext.py \
	--batch_size 1 \
	--skip_batch_num 2 \
	--infer_model_path "/home/lidanqin/models/se_resnext_50/129"\
	--test_file_list "/home/kbinias/data/imagenet/val_list.txt" \
	--data_dir '/home/kbinias/data/imagenet'\
	--profile \
	--use_transpiler True 
	


#!/bin/bash

if [ "$HOSTNAME" == "tomoplata-OptiPlex-790" ]; then
	VIRTUALENV_NAME=pytorch3
	# DATAROOT=~/data/geo_data_pkl
	HOME=~
	OPTIONS="--gpu_ids -1 --display_id 0"
else
	VIRTUALENV_NAME=cyclegan3
	# DATAROOT=/storage/Datasets/Geology-NicolasColtice/DS2-1810-RAW-DAT
	# DATAROOT=/storage/Datasets/Geology-NicolasColtice/pytorch_records_new_thresh
	DATAROOT=/storage/Datasets/Geology-NicolasColtice/old_pytorch_records_cont
	HOME=/home/tgillooly/

	source find_free_port.sh

	OPTIONS="--display_port $DISPLAY_PORT"
	VISDOM_OPTIONS="-p $DISPLAY_PORT"
	echo "Display port = $DISPLAY_PORT"
fi

if [ "$HOSTNAME" == "marky" ]; then
	VIRTUALENV_NAME=pytorch3_cuda8
fi

source $HOME/$VIRTUALENV_NAME/bin/activate

python -m visdom.server $VISDOM_OPTIONS > visdom.log 2>&1 &

	# --high_iter 25 --low_iter 5 \
	# --continue_train --which_epoch 55 --epoch_count 56 \
python train.py --dataroot $DATAROOT --name geo_self_attn_old_continents --x_size 512 \
        --model div_inline --which_direction BtoA \
        --niter 8000 --niter_decay 2000 \
	--num_discrims 1 --which_model_netD self-attn --use_hinge \
	--no_lsgan --norm batch --init_type orthogonal \
	--diff_in_numerator \
	--local_loss \
        --continent_data \
	--input_nc 3 --output_nc 1 \
        --g_lr 0.0004 --d_lr 0.0001 \
	--lambda_A 0.1 --lambda_B 0.005 --lambda_D 100 \
	--which_model_netG unet_256 \
	--display_freq 10 --print_freq 10 --save_epoch_freq 50 \
	--pool_size 0 --no_html --batchSize 5 --nThreads 2 $OPTIONS

#kill %1

#deactivate

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
	DATAROOT=/storage/Datasets/Geology-NicolasColtice/ellipses3
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
python train.py --dataroot $DATAROOT --name ellipse_critic_patchgan_orthogonal \
	--model div_inline --which_direction BtoA \
	--num_discrims 1 --which_model_netD n_layers --n_layers_D 4 \
	--no_lsgan --norm batch --init_type orthogonal \
	--high_iter 1 --low_iter 1 \
	--diff_in_numerator \
	--local_loss \
	--input_nc 3 --output_nc 1 \
        --g_lr 0.0004 --d_lr 0.0001 \
	--lambda_A 0.01 --lambda_D 100 \
	--which_model_netG unet_256 \
	--display_freq 10 --print_freq 10 \
	--pool_size 0 --no_html --batchSize 5 --nThreads 2 $OPTIONS

kill %1

deactivate

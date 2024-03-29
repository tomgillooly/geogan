#!/bin/bash

git checkout $GITBRANCH

if [ "$HOSTNAME" == "tomoplata-OptiPlex-790" ]; then
	VIRTUALENV_NAME=pytorch3
	DATAROOT=~/data/geology/new_data
	HOME=~
	OPTIONS="--gpu_ids -1 --display_id 0"
else
	VIRTUALENV_NAME=cyclegan3
	# DATAROOT=/storage/Datasets/Geology-NicolasColtice/DS2-1810-RAW-DAT
	DATAROOT=/storage/Datasets/Geology-NicolasColtice/pytorch_records
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

	# --continue_train --which_epoch 100 --epoch_count 101 --niter 300 --niter_decay 100 \
python train.py --dataroot $DATAROOT --name wgan_long_iter_base \
	--model pix2pix_geo --which_direction BtoA \
	--dataset_mode geo --no_lsgan --norm batch \
	--input_nc 3 --output_nc 3 \
	--high_iter 500 --low_iter 50 \
	--lambda_A 0.01 --lambda_B 100 --lambda_D 100 \
	--discrete_only \
	--num_discrims 1 \
	--which_model_netG unet_256 \
	--pool_size 0 --no_html --div_threshold 1000 --batchSize 4 $OPTIONS

kill %1

deactivate

git checkout dummy

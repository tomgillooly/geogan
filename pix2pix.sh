#!/bin/bash

if [ "$HOSTNAME" == "tomoplata-OptiPlex-790" ]; then
	VIRTUALENV_NAME=pytorch3
	DATAROOT=~/data/geology
	HOME=~
	OPTIONS="--gpu_ids -1 --display_id 0"
else
	VIRTUALENV_NAME=cyclegan3
	DATAROOT=/storage/Datasets/Geology-NicolasColtice/DS2-1810-RAW-DAT
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

python train.py --dataroot $DATAROOT --name geo_pix2pix_wgan_local_loss --model pix2pix_geo --which_model_netG unet_256 --which_direction BtoA --num_discrims 1 --local_loss \
	--input_nc 3 --output_nc 3 --lambda_A 100 --lambda_B 100 --dataset_mode geo --no_lsgan --norm batch --pool_size 0 --no_html --div_threshold 1000 $OPTIONS --batchSize 4

kill %1

deactivate

git checkout dummy

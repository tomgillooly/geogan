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
	# OPTIONS="--inpaint_file_dir ."
fi

source $HOME/$VIRTUALENV_NAME/bin/activate

python -m visdom.server > visdom.log 2>&1 &

python train.py --dataroot $DATAROOT --name geo_pix2pix_wgan --model pix2pix_geo --which_model_netG unet_256 --which_direction BtoA --input_nc 3 --output_nc 3 --lambda_A 100 --lambda_B 100 --dataset_mode geo --no_lsgan --norm batch --pool_size 0 --no_html --div_threshold 1000 $OPTIONS

kill %1

deactivate

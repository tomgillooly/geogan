#!/bin/bash

git checkout $GITBRANCH

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

	# --continue_train --which_epoch latest --epoch_count 45 \
python train.py --dataroot $DATAROOT --name base_autoencoder \
	--model pix2pix_geo --which_model_netG unet_256 --which_direction BtoA \
	--high_iter 25 --low_iter 5 --weighted_ce \
	--num_discrims 0 --which_model_netD wgan-gp --input_nc 3 --output_nc 3 \
	--lambda_A 100 --lambda_B 100 --dataset_mode geo --no_lsgan --norm batch \
	--pool_size 0 --no_html --div_threshold 1000 --batchSize 4 $OPTIONS

kill %1

deactivate

git checkout dummy

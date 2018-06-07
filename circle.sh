#!/bin/bash

git checkout $GITBRANCH

if [ "$HOSTNAME" == "tomoplata-OptiPlex-790" ]; then
	VIRTUALENV_NAME=pytorch3
	# DATAROOT=~/data/geo_data_pkl
	HOME=~
	OPTIONS="--gpu_ids -1 --display_id 0"
else
	VIRTUALENV_NAME=cyclegan3
	# DATAROOT=/storage/Datasets/Geology-NicolasColtice/DS2-1810-RAW-DAT
	# DATAROOT=/storage/Datasets/Geology-NicolasColtice/pytorch_records_new_thresh
	DATAROOT=/storage/Datasets/Geology-NicolasColtice/circles
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
	# --high_iter 25 --low_iter 5 \
python train.py --dataroot $DATAROOT --name circle_div_local_loss \
	--model div_inline --which_direction BtoA \
	--num_discrims 0 --alpha 0 --local_loss \
	--no_lsgan --norm batch \
	--input_nc 3 --output_nc 1 \
	--lambda_A 1 --lambda_D 100 \
	--which_model_netG unet_256 \
	--pool_size 0 --no_html --batchSize 100 --nThreads 2 $OPTIONS

kill %1

deactivate

git checkout dummy

#!/bin/bash

if [ "$HOSTNAME" == "tomoplata-OptiPlex-790" ]; then
	VIRTUALENV_NAME=pytorch3
	DATAROOT=~/data/circle_data_non_filled
	HOME=~
	OPTIONS="--gpu_ids -1 --display_id 0"
else
	VIRTUALENV_NAME=cyclegan3
	DATAROOT=/storage/Datasets/Geology-NicolasColtice/ellipses
	HOME=/home/tgillooly/
	OPTIONS="--gpu_ids -1"
fi

if [ "$HOSTNAME" == "marky" ]; then
	VIRTUALENV_NAME=pytorch3_cuda8
fi

source $HOME/$VIRTUALENV_NAME/bin/activate

# python -m visdom.server > visdom.log 2>&1 &

# python test.py --dataroot /storage/Datasets/Geology-NicolasColtice/DS2-1810-RAW-DAT --name geo_pix2pix_skel_remove --model pix2pix --which_model_netG unet_256 --which_direction BtoA --dataset_mode geo --norm batch --process skeleton_remove_small_components
python test.py --dataroot $DATAROOT --name $1 --model div_inline --which_model_netG unet_256 \
	--which_epoch $2  --how_many 10 --serial_batches \
	--which_direction BtoA  --dataset_mode geo --norm batch --input_nc 3 --output_nc 1 $OPTIONS

# kill %1

deactivate

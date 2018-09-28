#!/bin/bash

VIRTUALENV_NAME=cyclegan3
DATAROOT=/storage/Datasets/Geology-NicolasColtice/$DATASET
HOME=/home/tgillooly/

source find_free_port.sh

VISDOM_OPTIONS="-p $DISPLAY_PORT"
echo "Display port = $DISPLAY_PORT"

if [ "$HOSTNAME" == "marky" ]; then
	VIRTUALENV_NAME=pytorch3_cuda8
fi

source $HOME/$VIRTUALENV_NAME/bin/activate

python -m visdom.server $VISDOM_OPTIONS > visdom.log 2>&1 &

	# --continue_train --which_epoch 55 --epoch_count 56 \
	# --num_discrims 1 --which_model_netD self-attn --use_hinge \
	# --with_BCE --log_BCE --log_L2 \
	# --diff_in_numerator \
python train.py --x_size 512 \
    --model div_inline --which_direction BtoA \
    --niter 8000 --niter_decay 2000 \
	--num_discrims 1 \
	--no_lsgan --norm batch --init_type orthogonal \
	--local_loss \
	--input_nc 3 --output_nc 1 \
    --g_lr 0.0004 --d_lr 0.0001 \
	--lambda_A 0.1 --lambda_B 0.005 --lambda_D 100 \
	--which_model_netG unet_256 \
	--display_freq 10 --print_freq 10 --save_epoch_freq 100 \
	--pool_size 0 --no_html --batchSize 5 --nThreads 2 --display_port $DISPLAY_PORT \
	$@

kill %1

deactivate

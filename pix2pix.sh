#!/bin/bash

source /home/tgillooly/cyclegan3/bin/activate

python -m visdom.server 2>&1 > visdom.log &

python train.py --dataroot /storage/Datasets/Geology-NicolasColtice/DS2-1810-RAW-DAT --name geo_pix2pix --process skeleton --model pix2pix --which_model_netG unet_256 --which_direction BtoA --lambda_A 100 --dataset_mode geo --no_lsgan --norm batch --pool_size 0 --no_html

kill %1

deactivate

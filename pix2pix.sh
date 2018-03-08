#!/bin/bash

source /home/tgillooly/cyclegan3/bin/activate
# source ~/pytorch3/bin/activate

python -m visdom.server > visdom.log 2>&1 &

python train.py --dataroot /storage/Datasets/Geology-NicolasColtice/DS2-1810-RAW-DAT --name geo_pix2pix_dual --model pix2pix_geo --which_model_netG unet_256 --which_direction BtoA --input_nc 1 --output_nc 1 --lambda_A 100 --lambda_B 100 --dataset_mode geo --no_lsgan --norm batch --pool_size 0 --no_html
# python train.py --dataroot ../geology/data --name geo_pix2pix_dual --model pix2pix_geo --which_model_netG unet_256 --which_direction BtoA --input_nc 1 --output_nc 1 --lambda_A 100 --lambda_B 100 --dataset_mode geo --no_lsgan --norm batch --pool_size 0 --no_html --gpu_ids -1

kill %1

deactivate

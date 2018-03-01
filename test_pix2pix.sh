#!/bin/bash

source /home/tgillooly/cyclegan3/bin/activate

python -m visdom.server > visdom.log 2>&1 &

python test.py --dataroot /storage/Datasets/Geology-NicolasColtice/DS2-1810-RAW-DAT --name geo_pix2pix --model pix2pix --which_model_netG unet_256 --which_direction BtoA --dataset_mode geo --norm batch --process threshold_remove_small_components

kill %1

deactivate

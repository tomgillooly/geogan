#!/bin/env python

import glob
import numpy as np
import os
import re

dataroot = '/storage/Datasets/Geology-NicolasColtice/DS2-1810-RAW-DAT/train'

DIV_paths = glob.glob(os.path.join(dataroot, '*_DIV.dat'))
Vx_paths = glob.glob(os.path.join(dataroot, '*_Vx.dat'))
Vy_paths = glob.glob(os.path.join(dataroot, '*_Vy.dat'))

DIV_paths = sorted(DIV_paths)
Vx_paths = sorted(Vx_paths)
Vy_paths = sorted(Vy_paths)

A_paths = list(zip(DIV_paths, Vx_paths, Vy_paths))

rows = 256
cols = 512
depth = 1

for A_path in A_paths:
	DIV_path, Vx_path, Vy_path = A_path

	series_number = re.search('serie(\d+)', DIV_path).group(1)
	assert(series_number in Vx_path)
	assert(series_number in Vy_path)

	# It is possible to do an interpolation here, but it's really slow
	# and ends up looking about the same
	for path in A_path:
	    with open(path) as file:
	        try:
	            data = list(map(float, file.read().split()))
	        except ValueError as ex:
	            print(path)
	            raise ex

	        if len(data) > rows*cols*depth:
	            assert(len(data) == rows*cols*3)

	            x = np.array([data[i] for i in range(0, len(data), 3)])
	            y = np.array([data[i] for i in range(1, len(data), 3)])
	            data = [data[i] for i in range(2, len(data), 3)]

	            #return x, y, data

	        np.array(data).reshape((rows, cols), order='C')

		print(series_number + " ok")

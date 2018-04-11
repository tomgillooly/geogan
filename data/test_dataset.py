#!/bin/env python

dataroot = '/storage/Datasets/Geology-NicolasColtice/DS2-1810-RAW/train'

DIV_paths = glob.glob(os.path.join(dataroot, '*_DIV.dat'))
Vx_paths = glob.glob(os.path.join(dataroot, '*_Vx.dat'))
Vy_paths = glob.glob(os.path.join(dataroot, '*_Vy.dat'))

DIV_paths = sorted(DIV_paths)
Vx_paths = sorted(Vx_paths)
Vy_paths = sorted(Vy_paths)

A_paths = list(zip(DIV_paths, Vx_paths, Vy_paths))

for A_path in A_paths:
	DIV_path, Vx_path, Vy_path = A_path

	series_number = re.search('serie(\d+)', DIV_path).group(1)
	assert(series_number in Vx_path)
	assert(series_number in Vy_path)

	# It is possible to do an interpolation here, but it's really slow
	# and ends up looking about the same
	def read_geo_file(path):
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
	            data = np.array([data[i] for i in range(2, len(data), 3)]).reshape((rows, cols), order='C')

	            return x, y, data

	        return np.array(data).reshape((rows, cols), order='C')
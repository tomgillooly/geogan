import os
import pytest
import numpy as np
import shutil
import tempfile
import torch	

from data.geo_pickler import GeoPickler

import skimage.io as io

import matplotlib.pyplot as plt

@pytest.fixture
def fake_geo_data(num_series=2):
	test_data_dir = tempfile.mkdtemp()

	DIV_datas = []
	Vx_datas = []
	Vy_datas = []

	for series_number in range(num_series):
		# DIV_data = [
		# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		# [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
		# [0, 0, 0, 1, 0, 0, 1, 0, 0, 0],
		# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
		# [0, 1, 0, 0, 0, 0, 0, 0, 1, 0],
		# [0, 0, 1, 0, 0, 0, 0, 1, 0, 0],
		# [0, 0, 0, 1, 1, 1, 1, 0, 0, 0],
		# [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

		DIV_data = np.random.rand(9, 10)*20000
		Vx_data = np.random.rand(9, 10)*10000
		DIV_datas.append(DIV_data)
		Vx_datas.append(Vx_data)
		Vy_datas.append(Vx_data)

		Vx_data = [(x, y, data) for y, row in enumerate(Vx_data) for x, data in enumerate(row)]
		Vy_data = Vx_data


		DIV_data = [[x] for x in np.array(DIV_data).ravel()]


		for tag in ['DIV', 'Vx', 'Vy']:
			with open(os.path.join(test_data_dir, 'serie1{}_{}.dat'.format(series_number, tag)), 'w') as file:
				data_text = eval("'\\n'.join([' '.join([str(x) for x in data]) for data in " + tag + "_data])")
				# data_text = '\n'.join([','.join(data) for data in DIV_data])
				eval("file.write(data_text)")

	return test_data_dir, DIV_datas, Vx_datas, Vy_datas

def test_collects_files_in_dataroot():
	dataroot, _, _, _ = fake_geo_data(2)

	p = GeoPickler(dataroot)

	assert(p.num_folders == 0)
	
	p.collect_all()

	assert(p.num_folders == 1)

	dataroot, _, _, _ = fake_geo_data(3)

	p = GeoPickler(dataroot)

	assert(p.num_folders == 0)

	p.collect_all()

	assert(p.num_folders == 1)
	

def test_groups_by_series():
	dataroot, _, _, _ = fake_geo_data(3)

	p = GeoPickler(dataroot)

	p.collect_all()

	p.group_by_series()

	folder = p.get_folder_by_id(0)

	assert(len(folder) == 3)

	assert(p.get_series_in_folder(0) == [0, 1, 2])

	assert(folder[0] == ['serie10_DIV.dat', 'serie10_Vx.dat', 'serie10_Vy.dat'])
	assert(folder[1] == ['serie11_DIV.dat', 'serie11_Vx.dat', 'serie11_Vy.dat'])
	assert(folder[2] == ['serie12_DIV.dat', 'serie12_Vx.dat', 'serie12_Vy.dat'])


def test_searches_subfolders():
	dataroot, _, _, _ = fake_geo_data(1)
	subfolder, _, _, _ = fake_geo_data(2)

	shutil.move(subfolder, dataroot)

	p = GeoPickler(dataroot)

	p.collect_all()

	p.group_by_series()

	assert(list(p.folders.keys()) == ['', os.path.basename(subfolder)])

	assert(len(p.get_folder_by_id(0)) == 1)
	assert(len(p.get_folder_by_id(1)) == 2)


def test_build_data_dict(fake_geo_data):
	dataroot, DIV_datas, Vx_datas, Vy_datas = fake_geo_data
	
	p = GeoPickler(dataroot)

	p.collect_all()

	p.group_by_series()

	data_dict = p.get_data_dict(0, 0)

	assert((data_dict['A_DIV'] == DIV_datas[0]).all())
	assert((data_dict['A_Vx'] == Vx_datas[0]).all())
	assert((data_dict['A_Vy'] == Vy_datas[0]).all())


def test_build_one_hot(fake_geo_data):
	dataroot, DIV_datas, Vx_datas, Vy_datas = fake_geo_data
	
	p = GeoPickler(dataroot)

	p.collect_all()

	p.group_by_series()

	data_dict = p.get_data_dict(0, 0)

	DIV = (np.random.randn(*data_dict['A_DIV'].shape)*20000)

	data_dict['A_DIV'] = DIV

	p.create_one_hot(data_dict, 1000)

	one_hot = data_dict['A']

	assert([i in np.where(DIV > 1000) for i in np.where(one_hot[:, :, 0])])
	assert([i in np.where(np.logical_and(DIV < 1000, DIV < -1000)) for i in np.where(one_hot[:, :, 1])])
	assert([i in np.where(DIV < -1000) for i in np.where(one_hot[:, :, 2])])


def test_mask_location(fake_geo_data):
	dataroot, DIV_datas, Vx_datas, Vy_datas = fake_geo_data
	
	p = GeoPickler(dataroot)

	p.collect_all()

	p.group_by_series()

	data_dict = p.get_data_dict(0, 0)

	DIV = (np.random.randn(*data_dict['A_DIV'].shape)*20000)

	data_dict['A_DIV'] = DIV

	p.create_one_hot(data_dict, 1000)

	p.get_mask_loc(data_dict, 4, 6)

	one_hot = data_dict['A']

	mask_loc = data_dict['mask_locs']


	for x in range(one_hot.shape[1]-4):
		for y in range(one_hot.shape[0]-4):
			sum1 = np.sum(one_hot[y:y+4, x:x+4, 0])
			sum2 = np.sum(one_hot[y:y+4, x:x+4, 2])

			if (y, x) in mask_loc:
				assert(np.sum(one_hot[y:y+4, x:x+4, 0]) >= 6)
				assert(np.sum(one_hot[y:y+4, x:x+4, 2]) >= 6)
			else:
				assert(np.sum(one_hot[y:y+4, x:x+4, 0]) < 6 or np.sum(one_hot[y:y+4, x:x+4, 2]) < 6)


def test_normalises_continuous_data(fake_geo_data):
	dataroot, _, _, _ = fake_geo_data
	
	p = GeoPickler(dataroot)

	p.collect_all()

	p.group_by_series()

	data_dict = p.get_data_dict(0, 0)

	p.normalise_continuous_data(data_dict)

	assert(np.max(data_dict['A_DIV'].ravel()) == 1.0)
	assert(np.min(data_dict['A_DIV'].ravel()) == -1.0)

	assert(np.max(data_dict['A_Vx'].ravel()) == 1.0)
	assert(np.min(data_dict['A_Vx'].ravel()) == -1.0)

	assert(np.max(data_dict['A_Vy'].ravel()) == 1.0)
	assert(np.min(data_dict['A_Vy'].ravel()) == -1.0)


def test_pickling_contains_all_data(fake_geo_data, mocker):
	dataroot, _, _, _ = fake_geo_data
	
	p = GeoPickler(dataroot, 'out_dir')

	p.collect_all()

	p.group_by_series()

	mocker.patch('torch.save')

	p.pickle_series(0, 0, 4, 6)

	path = torch.save.call_args[0][1]
	data = torch.save.call_args[0][0]

	assert(path == 'out_dir/')
	assert(all([key in data.keys() for key in ['A', 'A_DIV', 'A_Vx', 'A_Vy', 'mask_locs', 'folder_name', 'series_number']]))
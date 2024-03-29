import glob
import os
import pytest
import random
import shutil
import skimage.io as io
import torch
import tempfile

import matplotlib.pyplot as plt
import numpy as np

from collections import namedtuple
from data.geo_dataset import GeoDataset, DataGenException, get_series_number

class NullOptions(object):
	pass

@pytest.fixture(scope='module')
def dataset(pytestconfig):
	opt = NullOptions()
	opt.dataroot=os.path.expanduser(pytestconfig.option.dataroot)
	opt.phase='test'
		# 
	opt.resize_or_crop='resize_and_crop'
		
	opt.loadSize=256
	opt.fineSize=256
	opt.which_direction='AtoB'
	    
	opt.input_nc=1
	opt.output_nc=1
	opt.no_flip=True
	opt.div_threshold=1000
	opt.inpaint_single_class=False
	    
	opt.continent_data=False


	geo = GeoDataset()
	geo.initialize(opt)

	return geo


def test_get_anything(dataset):
	assert(dataset[0] != None)


def test_series_number_order(dataset):
	assert(dataset[0]['series_number'] == 100001)
	assert(dataset[1]['series_number'] == 100002)


def test_mask_area(dataset):
	assert(dataset[0]['mask'].shape == (1, 256, 512))

	assert(torch.sum(dataset[0]['mask'].view(1, -1)) == 100*100)


def test_changing_mask_locations(dataset):
	#This permits some overlap, I think
	assert((dataset[0]['mask'].view(1, -1) != dataset[1]['mask'].view(1, -1)).any())
	assert((dataset[0]['mask'].view(1, -1) != dataset[2]['mask'].view(1, -1)).any())
	assert((dataset[1]['mask'].view(1, -1) != dataset[2]['mask'].view(1, -1)).any())

def test_mask_location_is_random(pytestconfig):
	# First test we can force an identical location with random seed
	# This is just a check for the test case itself

	random.seed(0)
	geo = dataset(pytestconfig)

	old_mask_loc = (geo[0]['mask_x1'][0], geo[0]['mask_x1'][0])

	random.seed(0)
	geo = dataset(pytestconfig)

	new_mask_loc = (geo[0]['mask_x1'][0], geo[0]['mask_x1'][0])

	assert(old_mask_loc == new_mask_loc)

	# Now see whether the mask location is different
	random.seed(1)
	geo = dataset(pytestconfig)

	new_mask_loc = (geo[0]['mask_x1'][0], geo[0]['mask_x1'][0])

	assert(old_mask_loc != new_mask_loc)


def test_getitem_performs_random_mask_search_again(dataset):
	x1_old = dataset[0]['mask_x1'][0]
	x1_new = dataset[0]['mask_x1'][0]

	assert(x1_old != x1_new)


def test_dataloader_repeats_mask_search(dataset):
	dataloader = torch.utils.data.DataLoader(
	    dataset,
	    batch_size=1,
	    shuffle=False,
	    num_workers=1)

	old_s_nos = []
	old_x1s = []

	# Just doing it this way because this is how it's done in train.py
	for i, old_data in enumerate(dataloader):
		old_x1s.append(old_data['mask_x1'][0].item())
		old_s_nos.append(old_data['series_number'][0].item())

		# Just test first 4
		if i > 3:
			break

	new_s_nos = []
	new_x1s = []

	for i, new_data in enumerate(dataloader):
		new_x1s.append(new_data['mask_x1'][0].item())
		new_s_nos.append(new_data['series_number'][0].item())

		# Just test first 4
		if i > 3:
			break


	# Even if we're loading a series again, the
	# mask has been recomputed
	assert(old_s_nos == new_s_nos)
	assert(old_x1s != new_x1s)



def test_mask_x_y_locations(dataset):
	data = dataset[0]

	x1 = data['mask_x1'][0]
	x2 = data['mask_x2'][0]
	y1 = data['mask_y1'][0]
	y2 = data['mask_y2'][0]

	mask = data['mask']

	# Check upper and lower corners
	assert(mask[0, y1-1, x1-1] == 0)
	assert(mask[0, y1, x1] == 1)
	assert(mask[0, y2-1, x2-1] == 1)
	assert(mask[0, y2, x2] == 0)

	# Check area inside mask x y locations is correct
	assert(torch.sum(mask[0, y1:y2, x1:x2].contiguous().view(1, -1)) == 100*100)

	# Check entire mask
	assert(torch.sum(mask.view(1, -1)) == 100*100)


# Code seems to do this if there's a problem
# def test_mask_not_at_0_0(dataset):
# 	for i in range(10):
# 		x1 = dataset[i]['mask_x1'][0]
# 		x2 = dataset[i]['mask_x2'][0]
# 		y1 = dataset[i]['mask_y1'][0]
# 		y2 = dataset[i]['mask_y2'][0]
	
# 		assert(x1 != 0)
# 		assert(y1 != 0)


def test_B_is_A_with_region_replaced_discrete(dataset):
	data = dataset[0]

	A = data['A']
	B = data['B']
	mask = data['mask']

	assert((A.masked_select(~mask) == B.masked_select(~mask)).all())

	# Now go by layer and check they match up
	assert((A[0, :, :].masked_select(mask) - B[0, :, :].masked_select(mask) == A[0, :, :].masked_select(mask)).all())
	# On the plate layer, some will match
	assert(not (A[1, :, :].masked_select(mask) == B[1, :, :].masked_select(mask)).all())
	assert((A[2, :, :].masked_select(mask) - B[2, :, :].masked_select(mask) == A[2, :, :].masked_select(mask)).all())


# def test_B_is_A_with_region_replaced_continuous(dataset):
# 	mask = dataset[0]['mask']

# 	A_DIV = dataset[0]['A_DIV']
# 	B_DIV = dataset[0]['B_DIV']
	
# 	# Area outside mask should be equal
# 	assert((A_DIV.masked_select(~mask) == B_DIV.masked_select(~mask)).all())

# 	# Now go by layer and check they match up
# 	assert((A_DIV[0, :, :].masked_select(mask) - B_DIV[0, :, :].masked_select(mask) == A_DIV[0, :, :].masked_select(mask)).all())
# 	# On the plate layer, some will match
# 	assert(not (A_DIV[1, :, :].masked_select(mask) == B_DIV[1, :, :].masked_select(mask)).all())
# 	assert((A_DIV[2, :, :].masked_select(mask) - B_DIV[2, :, :].masked_select(mask) == A_DIV[2, :, :].masked_select(mask)).all())

# 	A_Vx = dataset[0]['A_Vx']
# 	B_Vx = dataset[0]['B_Vx']
	
# 	# Area outside mask should be equal
# 	assert((A_Vx.masked_select(~mask) == B_Vx.masked_select(~mask)).all())

# 	# Now go by layer and check they match up
# 	assert((A_Vx[0, :, :].masked_select(mask) - B_Vx[0, :, :].masked_select(mask) == A_Vx[0, :, :].masked_select(mask)).all())
# 	# On the plate layer, some will match
# 	assert(not (A_Vx[1, :, :].masked_select(mask) == B_Vx[1, :, :].masked_select(mask)).all())
# 	assert((A_Vx[2, :, :].masked_select(mask) - B_Vx[2, :, :].masked_select(mask) == A_Vx[2, :, :].masked_select(mask)).all())

# 	A_Vy = dataset[0]['A_Vy']
# 	B_Vy = dataset[0]['B_Vy']
	
# 	# Area outside mask should be equal
# 	assert((A_Vy.masked_select(~mask) == B_Vy.masked_select(~mask)).all())

# 	# Now go by layer and check they match up
# 	assert((A_Vy[0, :, :].masked_select(mask) - B_Vy[0, :, :].masked_select(mask) == A_Vy[0, :, :].masked_select(mask)).all())
# 	# On the plate layer, some will match
# 	assert(not (A_Vy[1, :, :].masked_select(mask) == B_Vy[1, :, :].masked_select(mask)).all())
# 	assert((A_Vy[2, :, :].masked_select(mask) - B_Vy[2, :, :].masked_select(mask) == A_Vy[2, :, :].masked_select(mask)).all())


def test_discrete_data_is_one_hot(dataset):
	data = dataset[0]

	A = data['A']
	B = data['B']

	# Remember first axis is channels
	assert((torch.sum(A, dim=0) == torch.ones(*A.shape)).all())
	assert((torch.sum(B, dim=0) == torch.ones(*B.shape)).all())


def test_continuous_data_is_normalised(dataset):
	DIV = dataset[0]['A_DIV']
	Vx = dataset[0]['A_Vx']
	Vy = dataset[0]['A_Vy']

	assert(torch.max(DIV) <= 1)
	assert(torch.min(DIV) >= -1)

	# There might be a better way to test that it's image norm-ing, but
	# this will do for now
	assert(len(np.where(DIV.numpy() == 1)) > 0)
	assert(len(np.where(DIV.numpy() == -1)) > 0)

	assert(torch.max(Vx) <= 1)
	assert(torch.min(Vx) >= -1)
	assert(len(np.where(Vx.numpy() == 1)) > 0)
	assert(len(np.where(Vx.numpy() == -1)) > 0)

	assert(torch.max(Vy) <= 1)
	assert(torch.min(Vy) >= -1)
	assert(len(np.where(Vy.numpy() == 1)) > 0)
	assert(len(np.where(Vy.numpy() == -1)) > 0)


@pytest.fixture
def temp_dataset(dataset, folder_nums=[1, 2, 3, 4]):
	dataroot = dataset.opt.dataroot

	# Create a temporary directory to test
	temp_data_parent = tempfile.mkdtemp(dir='/tmp')

	folder_labels = ['{:02}'.format(num) for num in folder_nums]

	temp_data_dir_1 = os.path.join(temp_data_parent, folder_labels[0])
	temp_data_dir_2 = os.path.join(temp_data_parent, folder_labels[1])
	temp_data_dir_3 = os.path.join(temp_data_parent, folder_labels[2])
	temp_data_dir_4 = os.path.join(temp_data_parent, folder_labels[3])

	os.mkdir(temp_data_dir_1)
	os.mkdir(temp_data_dir_2)
	os.mkdir(temp_data_dir_3)
	os.mkdir(temp_data_dir_4)

	[shutil.copy(file, temp_data_dir_1) for file in glob.glob(dataroot + '/test/serie100001_project_*.dat')]
	[shutil.copy(file, temp_data_dir_2) for file in glob.glob(dataroot + '/test/serie100002_project_*.dat')]
	[shutil.copy(file, temp_data_dir_3) for file in glob.glob(dataroot + '/test/serie100003_project_*.dat')]
	[shutil.copy(file, temp_data_dir_3) for file in glob.glob(dataroot + '/test/serie100004_project_*.dat')]
	[shutil.copy(file, temp_data_dir_4) for file in glob.glob(dataroot + '/test/serie100004_project_*.dat')]
	[shutil.copy(file, temp_data_parent) for file in glob.glob(dataroot + '/test/serie100004_project_*.dat')]

	for folder in [temp_data_parent, temp_data_dir_1, temp_data_dir_2, temp_data_dir_3, temp_data_dir_4]:
		for tag in ['DIV', 'Vx', 'Vy']:
			with open(os.path.join(folder, tag + '_norm.dat'), 'w') as file:
				file.write('-10000 10000')

	# Check they're in the target directory
	glob.glob(os.path.join(temp_data_parent, '*.dat'))[0]
	glob.glob(os.path.join(temp_data_dir_1, '*.dat'))[0]
	glob.glob(os.path.join(temp_data_dir_2, '*.dat'))[0]
	glob.glob(os.path.join(temp_data_dir_3, '*.dat'))[0]
	glob.glob(os.path.join(temp_data_dir_4, '*.dat'))[0]

	# Now build a second dataset using this dummy directory
	opt = NullOptions()
	opt.dataroot=temp_data_parent
	opt.phase=''
		
	opt.resize_or_crop='resize_and_crop'
	    
	opt.loadSize=256
	opt.fineSize=256
	opt.which_direction='AtoB'
	    
	opt.input_nc=1
	opt.output_nc=1
	opt.no_flip=True
	opt.div_threshold=1000
	opt.inpaint_single_class=False
	    
	opt.continent_data=False
	
	geo = GeoDataset()
	geo.initialize(opt)

	div_files, vx_files, vy_files, _ = geo.get_dat_files(temp_data_parent)

	assert(len(div_files) == 6)
	assert(len(vx_files) == 6)
	assert(len(vy_files) == 6)

	geo.initialize(opt)

	return geo, opt

def test_directory_name_is_prepended_in_image_path(temp_dataset):
	geo, opt = temp_dataset

	# print(geo[0]['A_paths'])
	# Folders get flattened out
	# Sorted by directory, THEN by series number!

	assert(geo[0]['A_paths'] == os.path.join(geo.opt.dataroot, 'serie_100004'))
	assert(geo[1]['A_paths'] == os.path.join(geo.opt.dataroot, 'serie_01_100001'))
	assert(geo[2]['A_paths'] == os.path.join(geo.opt.dataroot, 'serie_02_100002'))
	assert(geo[3]['A_paths'] == os.path.join(geo.opt.dataroot, 'serie_03_100003'))

	# Sorted by directory, THEN by series number!
	assert(geo[4]['A_paths'] == os.path.join(geo.opt.dataroot, 'serie_03_100004'))
	assert(geo[5]['A_paths'] == os.path.join(geo.opt.dataroot, 'serie_04_100004'))


def test_folder_id(temp_dataset):
	geo, opt = temp_dataset
	
	assert(geo[0]['folder_id'] == 0)
	assert(geo[1]['folder_id'] == 1)
	assert(geo[2]['folder_id'] == 2)
	assert(geo[3]['folder_id'] == 3)
	assert(geo[4]['folder_id'] == 3)
	assert(geo[5]['folder_id'] == 4)

	# Test that object is modified outside of Dataset object
	assert(opt.num_folders == 5)


def test_return_sorted_folder_id(dataset):
	geo, opt = temp_dataset(dataset, folder_nums=[3, 6, 9, 12])
	
	assert(geo[0]['folder_id'] == 0)
	assert(geo[1]['folder_id'] == 1)
	assert(geo[2]['folder_id'] == 2)
	assert(geo[3]['folder_id'] == 3)
	assert(geo[4]['folder_id'] == 3)
	assert(geo[5]['folder_id'] == 4)

	# Test that object is modified outside of Dataset object
	assert(opt.num_folders == 5)


def test_no_continent_data_by_default(dataset):
	data = dataset[0]
	assert(not 'continents' in data.keys())


@pytest.fixture(scope='module')
def new_dataset():
	opt = NullOptions()
	opt.dataroot='test_data/with_continents'
	opt.phase=''
		
	opt.resize_or_crop='resize_and_crop'
	    
	opt.loadSize=256
	opt.fineSize=256
	opt.which_direction='AtoB'
	    
	opt.input_nc=1
	opt.output_nc=1
	opt.no_flip=True,
	opt.div_threshold=1000
	opt.inpaint_single_class=False
	    
	opt.continent_data=True

	geo = GeoDataset()
	geo.initialize(opt)

	return geo


def test_default_continent_map_is_blank():
	opt = NullOptions()
	opt.dataroot='test_data/no_continents'
	opt.phase=''
		
	opt.resize_or_crop='resize_and_crop'
	    
	opt.loadSize=256
	opt.fineSize=256
	opt.which_direction='AtoB'
	    
	opt.input_nc=1
	opt.output_nc=1
	opt.no_flip=True,
	opt.div_threshold=1000
	opt.inpaint_single_class=False
	    
	opt.continent_data=True

	geo = GeoDataset()
	geo.initialize(opt)

	data = geo[0]
	assert(torch.sum(data['continents']) == 0)


def test_handles_different_resolutions(new_dataset):
	assert(len(new_dataset) == 16)

	for i in range(len(new_dataset)):
		data = new_dataset[i]
		assert(data != None)

		assert(data['A'].shape == (3, 256, 512))
		assert(data['B'].shape == (3, 256, 512))


def test_zips_no_continent_data_correctly():
	opt = NullOptions()
	opt.dataroot=os.path.expanduser('~/data/new_geo_data')
	opt.phase=''
		
	opt.resize_or_crop='resize_and_crop'
	    
	opt.loadSize=256
	opt.fineSize=256
	opt.which_direction='AtoB'
	    
	opt.input_nc=1
	opt.output_nc=1
	opt.no_flip=True,
	opt.div_threshold=1000
	opt.inpaint_single_class=False
	    
	opt.continent_data=True

	geo = GeoDataset()
	geo.initialize(opt)

	# print(geo.A_paths)

	for paths in geo.A_paths:
		def get_subfolder(path):
			return os.path.basename(os.path.dirname(path))

		subfolders = [get_subfolder(path) for path in paths]
		s_nos = [get_series_number(path) for path in paths]

		assert(all([subfolder == subfolders[0] for subfolder in subfolders[1:]]))
		assert(all([s_no == s_nos[0] for s_no in s_nos[1:]]))


# def test_load_continent_data(new_dataset):
# 	import skimage.io as io
# 	# options_dict = dict(dataroot='test_data', phase='',
# 	# 	inpaint_file_dir='test_data', resize_or_crop='resize_and_crop',
# 	# 	# inpaint_file_dir=tempfile.mkdtemp(dir='/tmp'), resize_or_crop='resize_and_crop',
# 	#     loadSize=256, fineSize=256, which_direction='AtoB',
# 	#     input_nc=1, output_nc=1, no_flip=True, div_threshold=1000, inpaint_single_class=False,
# 	#     continent_data=True)
# 	# Options = namedtuple('Options', options_dict.keys())
# 	# opt = Options(*options_dict.values())

# 	# geo = GeoDataset()
# 	# geo.initialize(opt)
	
# 	# data = geo[0]
# 	for data in new_dataset:

# 		continents = data['continents']
		
# 		io.imshow(data['continents'].numpy().squeeze())
# 		io.show()

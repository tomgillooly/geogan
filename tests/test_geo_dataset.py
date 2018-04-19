import glob
import os
import pytest
import shutil
import torch
import tempfile

from collections import namedtuple
from data.geo_dataset import GeoDataset, get_dat_files

@pytest.fixture(scope='module')
def dataset(pytestconfig):
	# put together basic options class to pass to dataset builder
	inpaint_file_parent = tempfile.mkdtemp(dir='/tmp')
	inpaint_file_dir = os.path.join(inpaint_file_parent, 'test')

	os.mkdir(inpaint_file_dir)

	options_dict = dict(dataroot=os.path.expanduser(pytestconfig.option.dataroot), phase='test',
		# inpaint_file_dir=os.path.expanduser('~/data/geology/'), resize_or_crop='resize_and_crop',
		inpaint_file_dir=inpaint_file_parent, resize_or_crop='resize_and_crop',
	    loadSize=256, fineSize=256, which_direction='AtoB',
	    input_nc=1, output_nc=1, no_flip=True, div_threshold=1000, inpaint_single_class=False,
	    continent_data=False)
	Options = namedtuple('Options', options_dict.keys())
	opt = Options(*options_dict.values())


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


def test_mask_x_y_locations(dataset):
	x1 = dataset[0]['mask_x1'][0]
	x2 = dataset[0]['mask_x2'][0]
	y1 = dataset[0]['mask_y1'][0]
	y2 = dataset[0]['mask_y2'][0]

	mask = dataset[0]['mask']

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
def test_mask_not_at_0_0(dataset):
	for i in range(10):
		x1 = dataset[i]['mask_x1'][0]
		x2 = dataset[i]['mask_x2'][0]
		y1 = dataset[i]['mask_y1'][0]
		y2 = dataset[i]['mask_y2'][0]
	
		assert(x1 != 0)
		assert(y1 != 0)


def test_B_is_A_with_region_replaced_discrete(dataset):
	A = dataset[0]['A']
	B = dataset[0]['B']
	mask = dataset[0]['mask']

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
	A = dataset[0]['A']
	B = dataset[0]['B']

	# Remember first axis is channels
	assert((torch.sum(A, dim=0) == torch.ones(*A.shape)).all())
	assert((torch.sum(B, dim=0) == torch.ones(*B.shape)).all())


def test_continuous_data_is_normalised(dataset):
	DIV = dataset[0]['A_DIV']
	Vx = dataset[0]['A_Vx']
	Vy = dataset[0]['A_Vy']

	assert(torch.max(DIV) <= 1)
	assert(torch.min(DIV) >= -1)

	assert(torch.max(Vx) <= 1)
	assert(torch.min(Vx) >= -1)

	assert(torch.max(Vy) <= 1)
	assert(torch.min(Vy) >= -1)

def test_directory_name_is_prepended_in_image_path(dataset):
	dataroot = dataset.opt.dataroot

	# Create a temporary directory to test
	temp_data_parent = tempfile.mkdtemp(dir='/tmp')

	temp_data_dir_1 = os.path.join(temp_data_parent, '01')
	temp_data_dir_2 = os.path.join(temp_data_parent, '02')
	temp_data_dir_3 = os.path.join(temp_data_parent, '03')
	temp_data_dir_4 = os.path.join(temp_data_parent, '04')

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

	# Check they're in the target directory
	glob.glob(os.path.join(temp_data_parent, '*.dat'))[0]
	glob.glob(os.path.join(temp_data_dir_1, '*.dat'))[0]
	glob.glob(os.path.join(temp_data_dir_2, '*.dat'))[0]
	glob.glob(os.path.join(temp_data_dir_3, '*.dat'))[0]
	glob.glob(os.path.join(temp_data_dir_4, '*.dat'))[0]

	div_files, vx_files, vy_files, _ = get_dat_files(temp_data_parent)

	assert(len(div_files) == 6)
	assert(len(vx_files) == 6)
	assert(len(vy_files) == 6)

	# Now build a second dataset using this dummy directory
	options_dict = dict(dataroot=temp_data_parent, phase='',
		inpaint_file_dir=temp_data_parent, resize_or_crop='resize_and_crop',
	    loadSize=256, fineSize=256, which_direction='AtoB',
	    input_nc=1, output_nc=1, no_flip=True, div_threshold=1000, inpaint_single_class=False,
	    continent_data=False)
	Options = namedtuple('Options', options_dict.keys())
	opt = Options(*options_dict.values())

	geo = GeoDataset()
	geo.initialize(opt)

	# print(geo[0]['A_paths'])
	# Folders get flattened out
	# Sorted by directory, THEN by series number!

	assert(geo[0]['A_paths'] == os.path.join(temp_data_parent, 'serie_100004'))
	assert(geo[1]['A_paths'] == os.path.join(temp_data_parent, 'serie_01_100001'))
	assert(geo[2]['A_paths'] == os.path.join(temp_data_parent, 'serie_02_100002'))
	assert(geo[3]['A_paths'] == os.path.join(temp_data_parent, 'serie_03_100003'))

	# Sorted by directory, THEN by series number!
	assert(geo[4]['A_paths'] == os.path.join(temp_data_parent, 'serie_03_100004'))
	assert(geo[5]['A_paths'] == os.path.join(temp_data_parent, 'serie_04_100004'))


def test_no_continent_data_by_default(dataset):
	data = dataset[0]
	assert(not 'continents' in data.keys())


@pytest.fixture
def new_dataset():
	options_dict = dict(dataroot='test_data/with_continents', phase='',
		inpaint_file_dir='test_data/with_continents', resize_or_crop='resize_and_crop',
		# inpaint_file_dir=tempfile.mkdtemp(dir='/tmp'), resize_or_crop='resize_and_crop',
	    loadSize=256, fineSize=256, which_direction='AtoB',
	    input_nc=1, output_nc=1, no_flip=True, div_threshold=1000, inpaint_single_class=False,
	    continent_data=True)
	Options = namedtuple('Options', options_dict.keys())
	opt = Options(*options_dict.values())

	geo = GeoDataset()
	geo.initialize(opt)

	return geo


def test_handles_different_resolutions(new_dataset):
	assert(len(new_dataset) == 17)
	for i in range(len(new_dataset)):
		assert(new_dataset[i] != None)


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

import numpy as np
import os
import pytest
import random
import shutil
import tempfile
import torch
import torch.utils.data

from data.geo_unpickler import GeoUnpickler

@pytest.fixture
def fake_folder_hierarchy():
	test_data_dir = tempfile.mkdtemp()
	
	subfolder1 = tempfile.mkdtemp()
	subfolder2 = tempfile.mkdtemp()
	subfolder3 = tempfile.mkdtemp()

	for i in range(3):
		open(os.path.join(test_data_dir, '{:05}.pkl'.format(i)), 'w').close()
		open(os.path.join(subfolder1, '{:05}.pkl'.format(i)), 'w').close()
		open(os.path.join(subfolder2, '{:05}.pkl'.format(i)), 'w').close()
		open(os.path.join(subfolder3, '{:05}.pkl'.format(i)), 'w').close()

	shutil.move(subfolder1, test_data_dir)
	shutil.move(subfolder2, test_data_dir)
	shutil.move(subfolder3, test_data_dir)

	return test_data_dir


def test_finds_all_pkl_files(fake_folder_hierarchy):
	dataroot = fake_folder_hierarchy

	u = GeoUnpickler(dataroot)
	
	u.collect_all()

	assert(len(u.files) == 12)

	assert(len(u) == 12)


def test_pkl_files_in_series_order(fake_folder_hierarchy):
	# The folder older isn't so important
	dataroot = fake_folder_hierarchy

	u = GeoUnpickler(dataroot)
	
	u.collect_all()

	assert(len(u.files) == 12)

	series_numbers = [os.path.basename(file)[:-len('.pkl')] for file in u.files]

	assert(series_numbers == ['00000', '00001', '00002']*4)


def test_getitem_loads_pkl_file(fake_folder_hierarchy, mocker):
	dataroot = fake_folder_hierarchy

	u = GeoUnpickler(dataroot)
	
	u.collect_all()

	mocker.patch('torch.load')
	u.flip_images = mocker.MagicMock()
	u.create_masked_images = mocker.MagicMock()
	u.convert_to_tensor = mocker.MagicMock()

	data = u[0]

	path = torch.load.call_args[0][0]

	assert(path.startswith(dataroot))
	assert(path.endswith('00000.pkl'))

	data = u[1]

	path = torch.load.call_args[0][0]

	assert(path.startswith(dataroot))
	assert(path.endswith('00001.pkl'))


def test_create_mask_from_mask_locations(fake_folder_hierarchy, mocker):
	dataroot = fake_folder_hierarchy

	u = GeoUnpickler(dataroot)
	
	u.collect_all()

	fake_data = {
		'A': np.random.rand(100, 100, 3),
		'A_DIV': np.random.rand(100, 100),
		'A_Vx': np.random.rand(100, 100),
		'A_Vy': np.random.rand(100, 100),
		'mask_locs': [(10, 10)],
		'folder_name': 'fake_folder',
		'series_number': 0,
		'mask_size': 10,
		'min_num_mask_pixels': 10
		}

	u.create_mask(fake_data)

	assert('mask' in fake_data.keys())

	mask = fake_data['mask']
	x1 = fake_data['mask_x1']
	y1 = fake_data['mask_y1']
	x2 = fake_data['mask_x2']
	y2 = fake_data['mask_y2']

	assert x1 == 10
	assert y1 == 10
	assert x2 == 20
	assert y2 == 20

	assert(np.sum(mask.ravel()) == 10*10)

	assert(np.sum(mask[y1:y2, x1:x2].ravel()) == 10*10)


def test_generate_masked_images_from_mask_locations(fake_folder_hierarchy, mocker):
	dataroot = fake_folder_hierarchy

	u = GeoUnpickler(dataroot)
	
	u.collect_all()

	fake_data = {
		'A': np.random.rand(100, 100, 3),
		'A_DIV': np.random.rand(100, 100),
		'A_Vx': np.random.rand(100, 100),
		'A_Vy': np.random.rand(100, 100),
		'mask_locs': [(0, 0), (20, 20), (40, 40)],
		'folder_name': 'fake_folder',
		'series_number': 0,
		'mask_size': 10,
		'min_num_mask_pixels': 10
		}

	u.create_masked_images(fake_data)

	assert([key in fake_data.keys() for key in ['B', 'B_DIV', 'B_Vx', 'B_Vy']])

	mask = fake_data['mask']

	assert((fake_data['B'][:, :, 0][np.where(1.0 - mask)] == fake_data['A'][:, :, 0][np.where(1.0 - mask)]).all())
	assert((fake_data['B'][:, :, 0][np.where(mask)] == 0).all())

	assert((fake_data['B'][:, :, 1][np.where(1.0 - mask)] == fake_data['A'][:, :, 1][np.where(1.0 - mask)]).all())
	assert((fake_data['B'][:, :, 1][np.where(mask)] == 1).all())

	assert((fake_data['B'][:, :, 2][np.where(1.0 - mask)] == fake_data['A'][:, :, 2][np.where(1.0 - mask)]).all())
	assert((fake_data['B'][:, :, 2][np.where(mask)] == 0).all())

	assert((fake_data['B_DIV'][np.where(1.0 - mask)] == fake_data['A_DIV'][np.where(1.0 - mask)]).all())
	assert((fake_data['B_DIV'][np.where(mask)] == 0).all())

	assert((fake_data['B_Vx'][np.where(1.0 - mask)] == fake_data['A_Vx'][np.where(1.0 - mask)]).all())
	assert((fake_data['B_Vx'][np.where(mask)] == 0).all())
	
	assert((fake_data['B_Vy'][np.where(1.0 - mask)] == fake_data['A_Vy'][np.where(1.0 - mask)]).all())
	assert((fake_data['B_Vy'][np.where(mask)] == 0).all())


def test_data_tensor_conversion():
	u = GeoUnpickler('u')

	fake_data = {
		'A': np.random.rand(100, 100, 3),
		'A_DIV': np.random.rand(100, 100),
		'A_Vx': np.random.rand(100, 100),
		'A_Vy': np.random.rand(100, 100),
		'cont': np.random.rand(100, 100),
		'mask_locs': [(0, 0), (20, 20), (40, 40)],
		'folder_name': 'fake_folder',
		'series_number': 0,
		'mask_size': 10,
		'min_num_mask_pixels': 10
		}

	u.create_masked_images(fake_data)
	u.convert_to_tensor(fake_data)

	assert('torch.FloatTensor' in fake_data['A'].type())
	assert('torch.FloatTensor' in fake_data['A_DIV'].type())
	assert('torch.FloatTensor' in fake_data['A_Vx'].type())
	assert('torch.FloatTensor' in fake_data['A_Vy'].type())
	
	assert('torch.FloatTensor' in fake_data['B'].type())
	assert('torch.FloatTensor' in fake_data['B_DIV'].type())
	assert('torch.FloatTensor' in fake_data['B_Vx'].type())
	assert('torch.FloatTensor' in fake_data['B_Vy'].type())
	
	assert('torch.ByteTensor' in fake_data['mask'].type())
	assert('torch.LongTensor' in fake_data['mask_x1'].type())
	assert('torch.LongTensor' in fake_data['mask_y1'].type())
	assert('torch.LongTensor' in fake_data['mask_x2'].type())
	assert('torch.LongTensor' in fake_data['mask_y2'].type())

	assert('torch.ByteTensor' in fake_data['cont'].type())

	assert(fake_data['A'].shape == (3, 100, 100))
	assert(fake_data['A_DIV'].shape == (1, 100, 100))
	assert(fake_data['A_Vx'].shape == (1, 100, 100))
	assert(fake_data['A_Vy'].shape == (1, 100, 100))
	
	assert(fake_data['B'].shape == (3, 100, 100))
	assert(fake_data['B_DIV'].shape == (1, 100, 100))
	assert(fake_data['B_Vx'].shape == (1, 100, 100))
	assert(fake_data['B_Vy'].shape == (1, 100, 100))
	
	assert(fake_data['mask'].shape == (1, 100, 100))
	assert(fake_data['mask_x1'].shape == (1,))
	assert(fake_data['mask_y1'].shape == (1,))
	assert(fake_data['mask_x2'].shape == (1,))
	assert(fake_data['mask_y2'].shape == (1,))

	assert(fake_data['cont'].shape == (1, 100, 100))


def test_folder_id(fake_folder_hierarchy, mocker):
	dataroot = fake_folder_hierarchy

	u = GeoUnpickler(dataroot)
	
	u.collect_all()

	fake_data = {
		'A': np.random.rand(100, 100, 3),
		'A_DIV': np.random.rand(100, 100),
		'A_Vx': np.random.rand(100, 100),
		'A_Vy': np.random.rand(100, 100),
		'cont': np.random.rand(100, 100),
		'mask_locs': [(0, 0), (20, 20), (40, 40)],
		'folder_name': 'fake_folder',
		'series_number': 0,
		'mask_size': 10,
		'min_num_mask_pixels': 10
		}

	def empty_dictionary(*args, **kwargs):
		return {}

	mocker.patch('torch.load', side_effect=empty_dictionary)
	u.flip_images = mocker.MagicMock()
	u.create_masked_images = mocker.MagicMock()
	u.convert_to_tensor = mocker.MagicMock()

	all_data = [u[i] for i in range(len(u))]

	folder_ids = [data['folder_id'] for data in all_data]

	assert(all(np.unique(folder_ids) == [0, 1, 2, 3]))

	assert(len([folder_id for folder_id in folder_ids if folder_id == 0]) == 3)
	assert(len([folder_id for folder_id in folder_ids if folder_id == 1]) == 3)
	assert(len([folder_id for folder_id in folder_ids if folder_id == 2]) == 3)
	assert(len([folder_id for folder_id in folder_ids if folder_id == 3]) == 3)


class NullOptions(object):
	pass


def test_initialise(fake_folder_hierarchy):
	opt = NullOptions()

	opt.dataroot = fake_folder_hierarchy
	opt.inpaint_single_class = False

	u = GeoUnpickler()
	u.initialise(opt)

	assert(opt.num_folders == 4)


def test_single_inpaint(fake_folder_hierarchy, mocker):
	opt = NullOptions()

	opt.dataroot = fake_folder_hierarchy
	opt.inpaint_single_class = True

	u = GeoUnpickler()
	u.initialise(opt)

	fake_data = {
		'A': np.array([[[i]*3 for i in range(10)]] * 10),
		'A_DIV': np.array([list(range(10))] * 10),
		'mask_locs': [(10, 10)],
		'mask_size': 10
		}

	mocker.patch('random.random', return_value=0)

	u.create_masked_images(fake_data)

	assert([key in fake_data.keys() for key in ['B', 'B_DIV']])

	mask = fake_data['mask']

	assert((fake_data['B'][:, :, 0][np.where(1.0 - mask)] == fake_data['A'][:, :, 0][np.where(1.0 - mask)]).all())
	assert((fake_data['B'][:, :, 0][np.where(mask)] == 0).all())

	# Plate channel is original with ridge channel added in
	assert((fake_data['B'][:, :, 1][np.where(mask)] == fake_data['A'][:, :, 0][np.where(mask)] + fake_data['A'][:, :, 1][np.where(mask)]).all())
	
	# Other channel is left the same
	assert((fake_data['B'][:, :, 2] == fake_data['A'][:, :, 2]).all())
	
	mocker.patch('random.random', return_value=1)

	u.create_masked_images(fake_data)

	assert([key in fake_data.keys() for key in ['B', 'B_DIV']])

	mask = fake_data['mask']

	assert((fake_data['B'][:, :, 2][np.where(1.0 - mask)] == fake_data['A'][:, :, 2][np.where(1.0 - mask)]).all())
	assert((fake_data['B'][:, :, 2][np.where(mask)] == 0).all())

	# Plate channel is original with subduction channel added in
	assert((fake_data['B'][:, :, 1][np.where(mask)] == fake_data['A'][:, :, 0][np.where(mask)] + fake_data['A'][:, :, 1][np.where(mask)]).all())
	
	# Other channel is left the same
	assert((fake_data['B'][:, :, 0] == fake_data['A'][:, :, 0]).all())

	# Doesn't affect continuous classes
	assert((fake_data['B_DIV'][np.where(1.0 - mask)] == fake_data['A_DIV'][np.where(1.0 - mask)]).all())
	assert((fake_data['B_DIV'][np.where(mask)] == 0).all())


def test_random_flip(fake_folder_hierarchy, mocker):
	opt = NullOptions()

	opt.dataroot = fake_folder_hierarchy
	opt.inpaint_single_class = True

	u = GeoUnpickler()
	u.initialise(opt)

	fake_data = {
		'A': np.array([[[i]*3 for i in range(100)]] * 100),
		'A_DIV': np.array([list(range(100))] * 100),
		'mask_locs': [(10, 10)],
		'mask_size': 10
		}

	u.flip_images(fake_data)
	u.create_mask(fake_data)

	assert((fake_data['A'] == np.array([[[i]*3 for i in reversed(range(100))]] * 100)).all())
	assert((fake_data['A_DIV'] == np.array([list(reversed(range(100)))] * 100)).all())

	assert(fake_data['mask_x1'] == 80)
	assert(fake_data['mask_x2'] == 90)
	assert(fake_data['mask_y1'] == 10)
	assert(fake_data['mask_y2'] == 20)


def test_getitem_prepares_all(fake_folder_hierarchy, mocker):
	opt = NullOptions()

	opt.dataroot = fake_folder_hierarchy
	opt.inpaint_single_class = True
	opt.no_flip = False
	u = GeoUnpickler()
	u.initialise(opt)

	fake_data = mocker.MagicMock()
	mocker.patch('torch.load', return_value=fake_data)

	fs = mocker.patch.object(u, 'flip_images')
	cs = mocker.patch.object(u, 'create_masked_images')
	cr = mocker.patch.object(u, 'convert_to_tensor')

	manager = mocker.MagicMock()
	manager.attach_mock(fs, 'flip_images')
	manager.attach_mock(cs, 'create_masked_images')
	manager.attach_mock(cr, 'convert_to_tensor')

	mocker.patch('random.random', return_value=0)

	data = u[0]

	assert(manager.mock_calls == [mocker.call.flip_images(fake_data), 
			mocker.call.create_masked_images(fake_data), 
			mocker.call.convert_to_tensor(fake_data)])

	# Don't know how to clear manager, so just recreate
	manager = mocker.MagicMock()
	manager.attach_mock(fs, 'flip_images')
	manager.attach_mock(cs, 'create_masked_images')
	manager.attach_mock(cr, 'convert_to_tensor')

	mocker.patch('random.random', return_value=1)

	data = u[0]

	assert(manager.mock_calls == [
			mocker.call.create_masked_images(fake_data), 
			mocker.call.convert_to_tensor(fake_data)])



def test_can_build_dataloader_from_unpickler(fake_folder_hierarchy, mocker):
	opt = NullOptions()

	opt.dataroot = fake_folder_hierarchy
	opt.inpaint_single_class = True
	opt.no_flip = False

	def return_fake_data(*args, **kwargs):
		fake_data = {
			'A': np.random.rand(100, 100, 3),
			'A_DIV': np.random.rand(100, 100),
			'A_Vx': np.random.rand(100, 100),
			'A_Vy': np.random.rand(100, 100),
			'cont': np.random.rand(100, 100),
			'mask_locs': [(0, 0), (20, 20), (40, 40)],
			'folder_name': 'fake_folder',
			'series_number': 0,
			'mask_size': 10,
			'min_num_mask_pixels': 10
			}

		return fake_data

	mocker.patch('torch.load', side_effect=return_fake_data)

	u = GeoUnpickler()
	u.initialise(opt)

	data_loader = torch.utils.data.DataLoader(
            u,
            batch_size=1,
            shuffle=True,
            num_workers=1)

	dataset_size = len(data_loader)

	assert(dataset_size == 12)

	for data in data_loader:
		print("ok")


@pytest.mark.skip
def test_against_original_geo_dataset():
	pass






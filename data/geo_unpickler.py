import numpy as np
import os
import random
import torch

class DefaultOptions(object):
	pass

class GeoUnpickler(object):
	def __init__(self, dataroot=None, inpaint_single_class=False, no_flip=True):
		self.opt = DefaultOptions()
		self.opt.dataroot = dataroot
		self.opt.inpaint_single_class = inpaint_single_class
		self.opt.no_flip = no_flip
		self.opt.phase = ''

		self.files = []

		self.folder_id_lookup = {}
		self.num_folders = 0


	def initialise(self, opt):
		self.opt = opt

		self.collect_all()

		self.opt.num_folders = self.num_folders


	def collect_all(self):
		topdir = os.path.join(self.opt.dataroot, self.opt.phase).rstrip('/')

		for root, dirs, files in os.walk(topdir):
			self.files += sorted([os.path.join(root, file) for file in files if file.endswith('.pkl')])

			self.folder_id_lookup[root[len(topdir)+1:]] = self.num_folders
			self.num_folders += 1


	def create_mask(self, data_dict):
		mask_size = data_dict['mask_size']
		im_size = data_dict['A_DIV'].shape
		
		if 'mask_locs' in data_dict.keys():
			mask_loc = random.sample(data_dict['mask_locs'], 1)[0]
			# Remove these or it creates chaos when we're unpickling
			data_dict.pop('mask_locs')
		else:
			x_range = range(im_size[1] - mask_size - 1)
			y_range = range(im_size[0] - mask_size - 1)

			x = random.sample(x_range, 1)[0]
			y = random.sample(y_range, 1)[0]

			mask_loc = [y, x]

		mask = np.zeros(im_size)
		mask[mask_loc[0]:mask_loc[0]+mask_size, mask_loc[1]:mask_loc[1]+mask_size] = 1

		data_dict['mask'] = mask
		data_dict['mask_x1'] = mask_loc[1]
		data_dict['mask_y1'] = mask_loc[0]
		data_dict['mask_x2'] = mask_loc[1] + mask_size
		data_dict['mask_y2'] = mask_loc[0] + mask_size


	def create_masked_images(self, data_dict):
		if not 'mask' in data_dict.keys():
			self.create_mask(data_dict)

		mask = data_dict['mask']

		for tag in ['DIV', 'Vx', 'Vy', 'ResT']:
			if 'A_' + tag not in data_dict.keys():
				continue

			data_dict['B_' + tag] = data_dict['A_' + tag].copy()
			data_dict['B_' + tag][np.where(mask)] = 0

		B = data_dict['A'].copy()

		if self.opt.inpaint_single_class:
			# Layer to remove
			layer = int(round(random.random())*2)

			# Kill just that layer and the plate channel
			B[:, :, 1][np.where(mask)] = 0
			B[:, :, layer][np.where(mask)] = 0
		else:
			B[np.where(mask)] = [0, 0, 0]

		data_dict['B'] = B


	def flip_images(self, data_dict):
		for key in ['A', 'A_DIV', 'A_Vx', 'A_Vy', 'A_cont', 'A_ResT', 'B', 'B_DIV', 'B_Vx', 'B_Vy', 'B_ResT', 'mask', 'cont']:
			if not key in data_dict.keys():
				continue

			data_dict[key] = np.flip(data_dict[key], axis=1).copy()

		im_width = data_dict['A'].shape[1]

		if 'mask' in data_dict.keys():
			data_dict['mask_x1'] = im_width - data_dict['mask_x2']
			data_dict['mask_x2'] = data_dict['mask_x1'] + data_dict['mask_size']
		# If masks don't exist yet, redo mask locations so they'll be in the right place
		else:
			if 'mask_locs' in data_dict.keys():
				for i, (y, x) in enumerate(data_dict['mask_locs']):
					data_dict['mask_locs'][i] = (y, im_width - x - data_dict['mask_size'])


	def process_continents(self, data_dict):
		if 'cont' in data_dict.keys():
			return

		if 'A_cont' in data_dict.keys():			
			data_dict['cont'] = (data_dict['A_cont'] > 0).astype(np.uint8)
		else:
			data_dict['cont'] = np.zeros(data_dict['A_DIV'].shape, dtype=np.uint8)



	def convert_to_tensor(self, data_dict):
		for key in ['A', 'A_DIV', 'A_Vx', 'A_Vy', 'A_ResT', 'B', 'B_DIV', 'B_Vx', 'B_Vy', 'B_ResT', 'mask', 'cont']:
			if not key in data_dict.keys():
				continue

			item = data_dict[key]
			
			if len(item.shape) < 3:
				item = np.expand_dims(item, 2)

			item = item.transpose(2, 0, 1)

			if key == 'cont' or key == 'mask':
				item = torch.ByteTensor(item.copy())
			else:
				item = torch.FloatTensor(item.copy())

			data_dict[key] = item

		for tag in ['x1', 'x2', 'y1', 'y2']:
			key = 'mask_' + tag

			item = np.array([data_dict[key]])

			item = torch.LongTensor(item)

			data_dict[key] = item

		for key in ['DIV_max', 'DIV_min', 'DIV_thresh']:
			if not key in data_dict.keys():
				continue

			data_dict[key] = torch.FloatTensor([data_dict[key]])

		for key in ['conn_comp_hist']:
			if not key in data_dict.keys():
				continue

			data_dict[key] = torch.FloatTensor(data_dict[key])


	def __getitem__(self, idx):
		data = torch.load(self.files[idx])
		data['A'] = data['real_DISC']
		basedir = os.path.join(self.opt.dataroot, self.opt.phase).rstrip('/')
		
		data['folder_name'] = os.path.dirname(self.files[idx])[len(basedir)+1:]
		
		data['folder_id'] = self.folder_id_lookup[data['folder_name']]

		if (not self.opt.no_flip) and random.random() < 0.5:
			self.flip_images(data)
		
		self.create_masked_images(data)
		self.process_continents(data)

		self.convert_to_tensor(data)

		return data


	def __len__(self):
		return len(self.files)

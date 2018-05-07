import glob
import numpy as np
import os
import re
import torch

from collections import OrderedDict
from skimage.morphology import skeletonize
from scipy.signal import correlate2d

import matplotlib.pyplot as plt


class GeoPickler(object):
	def __init__(self, dataroot, out_dir=None):
		self.dataroot = dataroot
		self.out_dir = dataroot if not out_dir else out_dir
		self.num_series = 0

		self.folders = OrderedDict()
		self.num_folders = 0

		self.series = []

		self.filename_re = re.compile('serie1(?P<series_no>\d+)_*_(?P<tag>\w+).dat')


	def collect_all(self):
		topdir = self.dataroot.rstrip('/')

		for root, dirs, files in os.walk(topdir):
			folder = root[len(topdir)+1:]
			self.folders[folder] = files

		self.num_folders = len(self.folders)


	def get_folder_by_id(self, idx):
		return list(self.folders.values())[idx]


	def get_series_in_folder(self, idx):
		return list(self.get_folder_by_id(idx).keys())


	def group_by_series(self):
		for folder_name, files in self.folders.items():
			file_data = [self.filename_re.match(filename) for filename in files]

			series_numbers, lookup_idx = np.unique([int(match.group('series_no')) for match in file_data], return_inverse=True)
			
			num_series = len(series_numbers)

			self.folders[folder_name] = OrderedDict({idx: sorted([files[i] for i in np.where(lookup_idx == idx)[0]]) for idx in series_numbers})


	def read_geo_file(self, path):
	    data = OrderedDict()

	    with open(path) as file:
	        try:
	            lines = file.read().splitlines()

	            if len(lines[0].split()) == 1:
	                data['values'] = np.array([float(line) for line in lines])
	        
	            else:
	                data['x'] = np.array([float(line.split()[0]) for line in lines])
	                data['y'] = np.array([float(line.split()[1]) for line in lines])
	                data['values'] = np.array([float(line.split()[2]) for line in lines])

	        except ValueError as ex:
	            print(path)
	            raise ex

	    return data


	def get_data_dict(self, folder_id, series_no):
		files = self.get_folder_by_id(folder_id)[series_no]

		file_data = [self.filename_re.match(filename) for filename in files]

		tags = [match.group('tag') for match in file_data]

		folder_name = list(self.folders.keys())[folder_id]

		# Reconstruct filename
		def reconstruct_filename(folder_id, tag):
			return glob.glob(os.path.join(self.dataroot, folder_name, 'serie1' + str(series_no) + '*' + tag + '*'))[0]

		filenames = [reconstruct_filename(folder_id, tag) for tag in tags]

		series_data = [self.read_geo_file(filename) for filename in filenames]

		size_info = list(zip(*[(len(np.unique(data['x'])), len(np.unique(data['y']))) for data in series_data if 'x' in data.keys()]))

		assert(np.unique(size_info[0]) == size_info[0][0])
		assert(np.unique(size_info[1]) == size_info[1][0])

		cols = size_info[0][0]
		rows = size_info[1][0]

		data_dict = {'A_' + tag : data['values'].reshape(rows, cols) for tag, data in zip(tags, series_data)}
		data_dict['A_path'] = os.path.join(self.dataroot, )
		data_dict['folder_name'] = folder_name
		data_dict['series_number'] = series_no

		return data_dict


	def create_one_hot(self, data_dict, threshold):
		DIV_img = data_dict['A_DIV']
		ridge = skeletonize(DIV_img >= threshold).astype(float)
		subduction = skeletonize(DIV_img <= -threshold).astype(float)
		plate = np.ones(ridge.shape, dtype=float)
		plate[np.where(np.logical_or(ridge == 1, subduction == 1))] = 0

		data_dict['A'] = np.stack((ridge, plate, subduction), axis=2)


	def get_mask_loc(self, data_dict, mask_size, num_pixels):
		one_hot = data_dict['A']

		ridges = correlate2d(one_hot[:, :, 0] != 0, np.ones((mask_size, mask_size)), mode='valid')
		subductions = correlate2d(one_hot[:, :, 2] != 0, np.ones((mask_size, mask_size)), mode='valid')

		ridges = ridges >= num_pixels
		subductions = subductions >= num_pixels

		data_dict['mask_locs'] = list(zip(*np.where(np.logical_and(ridges, subductions))))


	def normalise_continuous_data(self, data_dict):
		for tag in ['DIV', 'Vx', 'Vy', 'ResT', 'cont']:
			key = 'A_' + tag
			if not key in data_dict.keys():
				continue

			data = data_dict[key]
			dmax = np.max(data.ravel())
			dmin = np.min(data.ravel())

			data = np.interp(data, [dmin, dmax], [-1, 1])

			data_dict[key] = data


	def pickle_series(self, folder_id, series_no, mask_size, num_pix_in_mask):
		data_dict = self.get_data_dict(folder_id, series_no)

		self.create_one_hot(data_dict, 1000)
		self.get_mask_loc(data_dict, mask_size, num_pix_in_mask)
		self.normalise_continuous_data(data_dict)

		torch.save(data_dict, os.path.join(self.out_dir, data_dict['folder_name']))
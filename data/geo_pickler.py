import glob
import numpy as np
import os
import re
import sys
import torch

from collections import OrderedDict
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize
from skimage.transform import resize
from scipy.signal import correlate2d


class GeoPickler(object):
	def __init__(self, dataroot, out_dir=None, row_height=None):
		self.dataroot = dataroot
		self.out_dir = dataroot if not out_dir else out_dir
		self.row_height = row_height

		self.num_series = 0

		self.folders = OrderedDict()
		self.num_folders = 0

		self.series = []

		self.filename_re = re.compile('serie1_?(?P<series_no>\d+)_(.*_)?(?P<tag>\w+).dat')


	def initialise(self):
		self.collect_all()
		self.group_by_series()


	def collect_all(self):
		topdir = self.dataroot.rstrip('/')

		for root, dirs, files in os.walk(topdir):
			folder = root[len(topdir)+1:]

			if len(files) > 0:
				self.folders[folder] = files

		self.num_folders = len(self.folders)


	def get_folder_by_id(self, idx):
		return list(self.folders.values())[idx]


	def get_series_in_folder(self, idx):
		return list(self.get_folder_by_id(idx).keys())


	def group_by_series(self):
		for folder_name, files in self.folders.items():
			file_data = [self.filename_re.match(filename) for filename in files]
			

			# Remove all files with no series number
			bad_format_idx = [i for i, match in enumerate(file_data) if match == None]
			[files.pop(idx-offset) for offset, idx in enumerate(bad_format_idx)]
			[file_data.pop(idx-offset) for offset, idx in enumerate(bad_format_idx)]

			self.folders[folder_name] = files

			series_numbers, lookup_idx = np.unique([int(match.group('series_no')) for match in file_data], return_inverse=True)
			
			num_series = len(series_numbers)

			self.folders[folder_name] = OrderedDict({s_no: sorted([files[i] for i in np.where(lookup_idx == idx)[0]]) for idx, s_no in enumerate(series_numbers)})


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
		filenames = [os.path.join(self.dataroot, folder_name, file) for file in files]

		series_data = [self.read_geo_file(filename) for filename in filenames]

		size_info = list(zip(*[(len(np.unique(data['x'])), len(np.unique(data['y']))) for data in series_data if 'x' in data.keys()]))

		assert(np.unique(size_info[0]) == size_info[0][0])
		assert(np.unique(size_info[1]) == size_info[1][0])

		cols = size_info[0][0]
		rows = size_info[1][0]

		# Skip, but this should be fixed later, by actually writing into a proper grid
		if any([len(data['values']) != rows*cols for data in series_data]):
			return {}

		data_dict = {'A_' + tag : data['values'].reshape(rows, cols) for tag, data in zip(tags, series_data)}

		if self.row_height:
			for key, im in data_dict.items():
				data_dict[key] = resize(im, (self.row_height, self.row_height * im.shape[1]/im.shape[0]), mode='constant')

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

		data_dict['DIV_thresh'] = threshold
		data_dict['A'] = np.stack((ridge, plate, subduction), axis=2)


	def get_mask_loc(self, data_dict, mask_size, num_pixels):
		one_hot = data_dict['A']

		ridges = correlate2d(one_hot[:, :, 0] != 0, np.ones((mask_size, mask_size)), mode='valid')
		subductions = correlate2d(one_hot[:, :, 2] != 0, np.ones((mask_size, mask_size)), mode='valid')

		ridges = ridges >= num_pixels
		subductions = subductions >= num_pixels

		data_dict['mask_locs'] = list(zip(*np.where(np.logical_and(ridges, subductions))))
		data_dict['mask_size'] = mask_size
		data_dict['min_pix_in_mask'] = num_pixels


	def normalise_continuous_data(self, data_dict):
		for tag in ['DIV', 'Vx', 'Vy', 'ResT']:
			key = 'A_' + tag
			if not key in data_dict.keys():
				continue

			data = data_dict[key]
			dmax = np.max(data.ravel())
			dmin = np.min(data.ravel())

			data_dict[tag + '_max'] = dmax
			data_dict[tag + '_min'] = dmin

			data = np.interp(data, [dmin, 0, dmax], [-1, 0, 1])

			# norm = max(abs(dmin), dmax)
			# data_dict['DIV_min'] = -norm
			# data_dict['DIV_max'] = norm

			# data = np.interp(data, [-norm, norm], [-1, 1])

			data_dict[key] = data


	def process_continents(self, data_dict):
		if 'A_cont' in data_dict.keys():
			data_dict['cont'] = (data_dict['A_cont'] > 0).astype(np.uint8)
		else:
			data_dict['cont'] = np.zeros(data_dict['A_DIV'].shape)


	def build_conn_comp_hist(self, data_dict):
		A = data_dict['A']

		r_label = label(A[:, :, 0])
		s_label = label(A[:, :, 2])

		r_area = [region.area for region in regionprops(r_label)]
		s_area = [region.area for region in regionprops(s_label)]

		r_area_hist, _ = np.histogram(r_area, bins=np.linspace(0, 256, 128))
		s_area_hist, bins = np.histogram(s_area, bins=np.linspace(0, 256, 128))
		total_hist = r_area_hist + s_area_hist

		data_dict['conn_comp_hist_ridge'] = r_area_hist
		data_dict['conn_comp_hist_subduction'] = s_area_hist
		data_dict['conn_comp_hist'] = total_hist


	def pickle_series(self, folder_id, series_no, threshold, mask_size, num_pix_in_mask):
		data_dict = self.get_data_dict(folder_id, series_no)

		if any([key not in data_dict.keys() for key in ['A_DIV', 'A_Vx', 'A_Vy']]):
			print('Missing DIV or velocity files, skipping')
			return

		self.normalise_continuous_data(data_dict)

		self.process_continents(data_dict)
		
		self.create_one_hot(data_dict, threshold)

		self.get_mask_loc(data_dict, mask_size, num_pix_in_mask)

		if len(data_dict['mask_locs']) == 0:
			print("Couldn't find any valid mask locations, skipping")
			return

		self.build_conn_comp_hist(data_dict)
		
		if not os.path.exists(os.path.join(self.out_dir, data_dict['folder_name'])):
			os.mkdir(os.path.join(self.out_dir, data_dict['folder_name']))

		torch.save(data_dict, os.path.join(self.out_dir, data_dict['folder_name'], '{:05}.pkl'.format(series_no)))

	def pickle_all(self, threshold, mask_size, num_pix_in_mask, verbose=False, skip_existing=False):
		for folder_id, folder_name in enumerate(self.folders.keys()):
			if verbose:
				print('Folder {} - {} of {}'.format(folder_name, folder_id+1, len(self.folders)))

			if isinstance(threshold, dict):
				if folder_name not in threshold.keys():
					print('No threshold defined for folder {}, skipping...'.format(folder_name))
					continue

				thresh = threshold[folder_name]
			elif isinstance(threshold, int):
				thresh = threshold
			else:
				raise Exception('Unrecognised threshold type')

			for count, series in enumerate(self.get_folder_by_id(folder_id).keys()):
				if verbose:
					sys.stdout.write('\rSeries {} of {}\t\t\t'.format(count+1, len(self.get_folder_by_id(folder_id))))
					sys.stdout.flush()
				if skip_existing and os.path.exists(os.path.join(self.out_dir, folder_name, '{:05}.pkl'.format(series))):
					continue

				self.pickle_series(folder_id, series, thresh, mask_size, num_pix_in_mask)
			print('')

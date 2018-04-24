#!/usr/bin/env python
import os

import skimage.io as io
import matplotlib.pyplot as plt

from collections import namedtuple

from data.geo_dataset import GeoDataset

from metrics.hausdorff import get_hausdorff

from random import sample

import numpy as np
import ot

from scipy.spatial.distance import euclidean


def get_em_distance(im1, im2, visualise=False):
	xs = np.where(im1)
	xs = np.array(list(zip(*xs)))

	xt = np.where(im2)
	xt = np.array(list(zip(*xt)))

	if not xs.any() or not xt.any():
		if visualise:
			return euclidean((0,0), xt.shape), np.zeros(xs.shape)
		else:
			return euclidean((0,0), xt.shape)

	M = ot.dist(xs, xt)
	M_norm = M / M.max()


	n = xs.shape[0]
	m = xt.shape[0]
	a, b = np.ones((n,)), np.ones((m,))
	G0 = ot.emd(a, b, M)

	# Summing along rows gives transport for xs->xt
	# Along columns is distance xt->xs
	dist = np.sum(np.multiply(G0, np.sqrt(M)).ravel())

	print(n)
	print(m)
	print(G0.shape)
	# print(G0.sum(axis=1))
	# print(G0.ravel().sum())
	print(np.where(G0[:, 0]))
	print(np.where(G0[0, :]))

	print(G0[:, 0][np.where(G0[:, 0])])
	print(G0[0, :][np.where(G0[0, :])])

	print(M[:5, :5])

	for i in range(5):
		for j in range(5):
			print(i, j, euclidean(xs[i], xt[j])**2)


	if visualise:
		fig = plt.figure(1)
		ot.plot.plot2D_samples_mat(np.vstack((xs[:, 1], 100-xs[:, 0])).T, np.vstack((xt[:, 1], 100-xt[:, 0])).T, G0, c=[0.5, 1, 0])
		plt.plot(xs[:, 1], 100-xs[:, 0], '+b', label='Source samples')
		plt.plot(xt[:, 1], 100-xt[:, 0], 'xr', label='Target samples')
		plt.xlim([0, 100])
		plt.ylim([0, 100])

		plt.title('OT matrix with samples')

		plt.savefig('/tmp/tempfig.png')
		buf = io.imread('/tmp/tempfig.png')

		plt.close(1)

		return dist, buf

	return dist


if __name__ == '__main__':

	options_dict = dict(dataroot=os.path.expanduser('~/data/geology/'), phase='test', inpaint_file_dir=os.path.expanduser('~/data/geology/'), resize_or_crop='resize_and_crop',
	    loadSize=256, fineSize=256, which_direction='AtoB', input_nc=1, output_nc=1, no_flip=True, div_threshold=1000, inpaint_single_class=False)
	Options = namedtuple('Options', options_dict.keys())
	opt = Options(*options_dict.values())
	geo = GeoDataset()
	geo.initialize(opt)

	for i in range(1):
		data = geo[i]

		B = data["A"]
		mask = data["mask"]

		inpaint_region = B.masked_select(mask.repeat(3, 1, 1)).numpy().reshape(3, 100, 100).transpose(1, 2, 0)

		d, im = get_em_distance(inpaint_region[:, :, 0], inpaint_region[:, :, 2], visualise=True)
		print(inpaint_region[:, :, 0].shape)
		print(inpaint_region[:, :, 2].shape)
		d2 = get_em_distance(inpaint_region[:, :, 2], inpaint_region[:, :, 0], visualise=False)

		print(d, d2)

		# fig, ax = plt.subplots(1, 2)
		# ax = ax.ravel()

		# ax[0].imshow(inpaint_region)
		# ax[1].imshow(im)
		# io.show()

		
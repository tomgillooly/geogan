import matplotlib.pyplot as plt
import numpy as np
import pytest
import skimage.io as io

from metrics.hausdorff import get_hausdorff, get_hausdorff_exc, get_pairs

@pytest.fixture
def images():
	X = np.zeros((100, 100))
	Y = np.zeros((100, 100))

	X[50, 20] = 1
	Y[50, 30] = 1

	assert(X.ravel().sum() == 1)
	assert(Y.ravel().sum() == 1)

	return X, Y


def test_one_point_each(images):
	X, Y = images

	d1, d2, d3 = get_hausdorff(X, Y)

	assert(d1 == d2)
	assert(d1 == d3)

	assert(d1 == 10)


def test_two_to_one_point(images):
	X, Y = images

	X[50, 10] = 1

	d1, d2, d3 = get_hausdorff(X, Y)

	assert(d1 == 15)
	assert(d2 == 10)


def test_exclusive_pairs():
	p1 = [(0, 0), (10, 10)]
	p2 = [(1, 1), (2, 2)]

	pairs = get_pairs(p1, p2)

	assert(pairs == [(0, 0), (1, 1)])

	p1 = [(10, 10), (0, 0)]
	p2 = [(1, 1), (2, 2)]

	pairs = get_pairs(p1, p2)

	# Closest match always comes first
	assert(pairs == [(1, 0), (0, 1)])


def test_not_ordered():
	p1 = [(0, 0), (10, 0)]
	p2 = [(-10, 0), (9, 0)]

	pairs = get_pairs(p1, p2)

	assert(pairs == [(1, 1), (0, 0)])


def test_unmatched_point():
	p1 = [(0, 0), (10, 10)]
	p2 = [(1, 1)]

	pairs = get_pairs(p1, p2)

	assert(pairs == [(0, 0), (1, None)])

	pairs = get_pairs(p2, p1)

	assert(pairs == [(0, 0)])

# def test_problem_image():
# 	mask_w = 386
# 	mask_h = 75

# 	im_gt = io.imread("results/geo_pix2pix_wgan_no_mask_to_critic/test_latest/images/serie100004_project_DIV_out_ground_truth_one_hot.png")[mask_h:mask_h+99, mask_w:mask_w+99, :]
# 	im_p = io.imread("results/geo_pix2pix_wgan_no_mask_to_critic/test_latest/images/serie100004_project_DIV_out_output_one_hot.png")[mask_h:mask_h+99, mask_w:mask_w+99, :]

# 	d1, d2, d3, im1, im2 = get_hausdorff_exc(im_p[:, :, 0], im_gt[:, :, 0], visualise=True)
	
# 	fig, ax = plt.subplots(1, 2)
# 	ax = ax.ravel()

# 	ax[0].imshow(im_p[:, :, 0])
# 	ax[1].imshow(im_gt[:, :, 0])
# 	# ax[2].imshow(im1)
# 	# ax[3].imshow(im2)
# 	io.show()

# 	assert(d1 != d2)
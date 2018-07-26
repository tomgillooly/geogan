import numpy as np
import matplotlib.pyplot as plt

from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment

def get_emd(im1, im2, visualise=False, im1_label='Predicted', im2_label='Actual'):
	im1_coords = np.array(np.where(im1)).T
	im2_coords = np.array(np.where(im2)).T
	
	y_size = im1.shape[0]
	x_size = im1.shape[1]
	max_distance = np.sqrt(x_size**2 + y_size**2)

	num_im1 = len(im1_coords)
	num_im2 = len(im2_coords)

	if abs(num_im1 - num_im2) > 50 and not visualise:
		return max_distance

	cost = cdist(im1_coords, im2_coords)
	# print('source, dest', cost.shape)
	diff = cost.shape[0] - cost.shape[1]
	axis = 0 if diff < 0 else 1
	diff = abs(diff)
	dummy_points = np.ones((diff, cost.shape[1]) if axis==0 else (cost.shape[0], diff)) * max_distance
	cost = np.concatenate((cost, dummy_points), axis=axis)

	source_ind, dest_ind = linear_sum_assignment(cost)

	average_distance = np.mean(cost[source_ind, dest_ind])

	source_inliers_ind = source_ind[:num_im1][np.where(dest_ind[:num_im1] < num_im2)]
	source_inliers = im1_coords[source_inliers_ind, :]
	source_outliers_ind = source_ind[:num_im1][np.where(dest_ind[:num_im1] >= num_im2)]
	source_outliers = im1_coords[source_outliers_ind, :]

	dest_inliers_ind = dest_ind[:num_im1][source_inliers_ind]
	dest_inliers = im2_coords[dest_inliers_ind, :]
	dest_outliers_ind = dest_ind[:num_im2][np.where(source_ind[:num_im2] >= num_im1)]
	dest_outliers = im2_coords[dest_outliers_ind, :]

	if visualise:
		plt.figure(1)
		for s, d in zip(source_inliers, dest_inliers):
			plt.plot([s[1], d[1]], [y_size-s[0], y_size-d[0]], c=[.5, .5, 1])

		plt.plot(source_inliers[:, 1], y_size-source_inliers[:, 0], '+b', label='Source samples')
		plt.plot(dest_inliers[:, 1], y_size-dest_inliers[:, 0], 'xr', label='Target samples')
		if len(source_outliers) > 0:
			plt.plot(source_outliers[:, 1], y_size-source_outliers[:, 0], '+g', label='Source Outliers')
		if len(dest_outliers) > 0:
			plt.plot(dest_outliers[:, 1], y_size-dest_outliers[:, 0], '+m', label='Dest Outliers')

		plt.legend(loc=0)
		
		plt.title('OT matrix with samples\nAverage distance: {:.02f}'.format(average_distance))
		# plt.axis('equal')
		plt.xlim([0, x_size])
		plt.ylim([0, y_size])
		plt.axes().set_aspect('equal')
		plt.grid()

		plt.savefig('/tmp/tmpimg.png')
		img = (plt.imread('/tmp/tmpimg.png')*255).astype(np.uint8)
		plt.close(1)

		return average_distance, img

	return average_distance

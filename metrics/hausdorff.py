import numpy as np

from scipy.spatial.distance import directed_hausdorff, cdist

import matplotlib.pyplot as plt
import skimage.io as io


def get_hausdorff(im1, im2, visualise=False, im1_label='Predicted', im2_label='Actual'):
    im1_coords = np.array(np.where(im1)).T
    im2_coords = np.array(np.where(im2)).T

    max_value = np.sqrt(im1.shape[0]**2 + im1.shape[1]**2)

    d_h_2to1, d_h_1to2, d_h_s = 0.0, 0.0, 0.0

    im1 = np.expand_dims(im1, 2)
    im2 = np.expand_dims(im2, 2)

    im1_stack = np.concatenate((im1, np.zeros((im1.shape[0], im1.shape[1], 2))), axis=2)
    im2_stack = np.concatenate((np.zeros((im2.shape[0], im2.shape[1], 2)), im2), axis=2)


    if visualise:
        fig = plt.figure(1)
        f1_im1_h = plt.scatter(im1_coords[:, 1], im1.shape[0] - im1_coords[:, 0], c='r', marker='x', label=im1_label)
        f1_im2_h = plt.scatter(im2_coords[:, 1], im2.shape[0] - im2_coords[:, 0], c='b', marker='+', label=im2_label)
        plt.legend(scatterpoints=1)

        fig = plt.figure(2)
        f2_im1_h = plt.scatter(im1_coords[:, 1], im1.shape[0] - im1_coords[:, 0], c='r', marker='x', label=im1_label)
        f2_im2_h = plt.scatter(im2_coords[:, 1], im2.shape[0] - im2_coords[:, 0], c='b', marker='+', label=im2_label)
        plt.legend(scatterpoints=1)


    if not im1.any() or not im2.any():
        d_h_1to2 = max_value
        d_h_2to1 = max_value
        d_h_s = max_value
    elif im1.any() and im2.any():
        D = cdist(im1_coords, im2_coords)

        closest_1to2_idx = np.argmin(D, axis=1)

        closest_2to1_idx = np.argmin(D, axis=0)

        d_h_1to2 = np.mean(np.min(D, axis=1))
        d_h_2to1 = np.mean(np.min(D, axis=0))
        # d_h_1to2, i1_1to2, i2_1to2 = directed_hausdorff(im1_coords, im2_coords)
        # d_h_2to1, i2_2to1, i1_2to1 = directed_hausdorff(im2_coords, im1_coords)

        # print(im1_coords[i1_1to2])
        # print(im2_coords[i2_1to2])

        # print(im1_coords[i1_2to1])
        # print(im2_coords[i2_2to1])

        # combined_1to2[im1_coords[i1_1to2][0], im1_coords[i1_1to2][1], :] = [0.5, 1, 0.5] + 0.5 * combined_1to2[im1_coords[i1_1to2][0], im1_coords[i1_1to2][1], :]
        # combined_1to2[im2_coords[i2_1to2][0], im2_coords[i2_1to2][1], :] = [0.5, 1, 0.5] + 0.5 * combined_1to2[im2_coords[i2_1to2][0], im2_coords[i2_1to2][1], :]

        # combined_2to1[im1_coords[i1_2to1][0], im1_coords[i1_2to1][1], :] = [0.5, 1, 0.5] + 0.5 * combined_2to1[im1_coords[i1_2to1][0], im1_coords[i1_2to1][1], :]
        # combined_2to1[im2_coords[i2_2to1][0], im2_coords[i2_2to1][1], :] = [0.5, 1, 0.5] + 0.5 * combined_2to1[im2_coords[i2_2to1][0], im2_coords[i2_2to1][1], :]
        # combined_1to2[0, 1, 2][im2_coords[i2_1to2]] = [1, 1, 1]
        
        # combined_2to1[0, 1, 2][im1_coords[i1_2to1]] = [1, 1, 1]
        # combined_2to1[0, 1, 2][im2_coords[i2_2to1]] = [1, 1, 1]

        d_h_s = max(d_h_1to2, d_h_2to1)

 
        if visualise:
            fig = plt.figure(1)
            
            for i, (y, x) in enumerate(im1_coords):
                closest = im2_coords[closest_1to2_idx[i]]
                plt.plot([x, closest[1]], [im1.shape[0] - p for p in [y, closest[0]]], color=[0, 1, 0])

            plt.title('Average distance: {:.02f}'.format(d_h_1to2))
            # plt.axis('equal')
            plt.xlim([0, 100])
            plt.ylim([0, 100])
            plt.axes().set_aspect('equal')
            plt.grid()
            
            fig = plt.figure(2)

            for i, (y, x) in enumerate(im2_coords):
                closest = im1_coords[closest_2to1_idx[i]]
                plt.plot([x, closest[1]], [im1.shape[0] - p for p in [y, closest[0]]], color=[0, 1, 0])

            plt.title('Average distance: {:.02f}'.format(d_h_2to1))
            # plt.axis('equal')
            plt.xlim([0, 100])
            plt.ylim([0, 100])
            plt.axes().set_aspect('equal')
            plt.grid()
            

    if visualise:
        fig = plt.figure(1)
        plt.savefig('/tmp/tmpimg.png')
        combined_1to2 = io.imread('/tmp/tmpimg.png')
        plt.close(1)

        fig = plt.figure(2)
        plt.savefig('/tmp/tmpimg.png')
        combined_2to1 = io.imread('/tmp/tmpimg.png')
        plt.close(2)


        return d_h_1to2, d_h_2to1, d_h_s, combined_1to2, combined_2to1
    else:
        return d_h_1to2, d_h_2to1, d_h_s
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

from collections import namedtuple, OrderedDict

from skimage.transform import resize
# import skimage.io as io
import numpy as np

# import matplotlib.pyplot as plt


# def threshold(img, threshold=5000):
#     zero_idx = np.where(np.abs(img) <= threshold)
#     #out_img[map(int, x[zero_idx]), map(int, y[zero_idx])] = 0
#     img[zero_idx] = 0.5

#     one_idx = np.where(img > threshold)
#     #out_img[map(int, x[one_idx]), map(int, y[one_idx])] = 1
#     img[one_idx] = 1

#     neg_one_idx = np.where(img < -threshold)
#     #out_img[map(int, x[neg_one_idx]), map(int, y[neg_one_idx])] = -1
#     img[neg_one_idx] = 0

#     return img


class GeoDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase)
        self.A_paths = []

        for root, dirs, files in os.walk(self.dir_A):
            self.A_paths += [os.path.join(root, file) for file in files]

        assert(opt.resize_or_crop == 'resize_and_crop')

    def __getitem__(self, index):
        A_path = self.A_paths[index]

        def format_correct(file_path):
            with open(file_path) as  file:
                line = file.readline()
                file.seek(0)

                return len(line.split()) == 1

        while not format_correct(A_path):
            self.A_paths.pop(index)
            A_path = self.A_paths[index]

        # imgnames=get_img_names(inputdir, max=0)

        # DEBUG : only use 2 images!
        # imgnames = [ imgnames[0], imgnames[1]]

        # N=len(imgnames)
        rows = 256
        cols = 512
        depth = 1
        # print('Writing', filename)

        #filename_queue = tf.train.string_input_producer(["test2.csv"])

        # inputs = torch.FloatTensor(N, 1, rows, cols).zero_()
        # targets = torch.FloatTensor(N, 1, rows, cols).zero_()

            # sys.stdout.write("Reading {} ({}/{})".format(imgn,index+1, N))

        # Read the image into an array
        A = np.empty([cols,rows], dtype=np.float32)
            # r=0
            # c=0
        with open(A_path) as file:
            data = list(map(float, file.read().split()))
            A = np.array(data).reshape((cols, rows), order='F')

        # A = Image.fromarray(A.astype(np.uint8))
        A = resize(A, (self.opt.loadSize * 2, self.opt.loadSize), mode='constant')

        w = A.shape[1]
        h = A.shape[0]
        
        w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        A = A[h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        
        def threshold(pixel):
            if pixel.shape:
                return list(map(threshold, pixel))

            if pixel < -1000:
                return 0
            elif pixel > 1000:
                return 1
            else:
                return 0.5

        B = list(map(threshold, A))
        B = np.array(B)*255
        
        A = np.interp(A, [np.min(A), np.max(A)], [0, 255])

        if len(A.shape) < 3:
            A = np.tile(A, (3, 1, 1)).transpose(1, 2, 0)
            B = np.tile(B, (3, 1, 1)).transpose(1, 2, 0)
        
        A = transforms.ToTensor()(A)
        B = transforms.ToTensor()(B)

        A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)

        if output_nc == 1:  # RGB to gray
            tmp = B[0, ...] * 0.299 + B[1, ...] * 0.587 + B[2, ...] * 0.114
            B = tmp.unsqueeze(0)

        return {'A': A, 'B': B,
                'A_paths': A_path, 'B_paths': os.path.join(os.path.splitext(A_path)[0], '_thresh', os.path.splitext(A_path)[1])}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'GeoDataset'


# if __name__ == '__main__':
# os.chdir('..')
# print(os.getcwd())
# options_dict = dict(dataroot='geology/data', phase='', resize_or_crop='resize_and_crop', loadSize=256, fineSize=256, which_direction='AtoB', input_nc=1, output_nc=1, no_flip=True)
# Options = namedtuple('Options', options_dict.keys())
# opt = Options(*options_dict.values())
# geo = GeoDataset()
# geo.initialize(opt)

# stuff = geo[1]
# plt.subplot(121)
# io.imshow(stuff['A'].numpy().transpose(1, 2, 0).squeeze())
# plt.subplot(122)
# io.imshow(stuff['B'].numpy().transpose(1, 2, 0).squeeze())
# plt.show()
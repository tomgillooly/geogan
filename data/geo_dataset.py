import glob
import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image

from collections import namedtuple, OrderedDict

from skimage.transform import resize
from skimage.morphology import skeletonize, label, remove_small_objects, remove_small_holes
from skimage.measure import regionprops
import numpy as np

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
        self.process = opt.process

        # for root, dirs, files in os.walk(self.dir_A):
        #     self.A_paths += [os.path.join(root, file) for file in files]

        # self.A_paths = [path for path in self.A_paths if path.endswith(".dat")]

        DIV_paths = glob.glob(os.path.join(self.dir_A, '*_DIV.dat'))
        Vx_paths = glob.glob(os.path.join(self.dir_A, '*_Vx.dat'))
        Vy_paths = glob.glob(os.path.join(self.dir_A, '*_Vy.dat'))

        # self.A_paths = sorted(self.A_paths)

        self.A_paths = list(zip(DIV_paths, Vx_paths, Vy_paths))
        
        assert(len(self.A_paths) > 0)

        assert(opt.resize_or_crop == 'resize_and_crop')

        self.inpaint_regions_file = os.path.join(opt.inpaint_file_dir, opt.phase, 'inpaint_regions')
        self.inpaint_regions = [None]*len(self.A_paths)
        # self.inpaint_regions_file = os.path.join(opt.phase, 'inpaint_regions')
        
        if os.path.exists(self.inpaint_regions_file):
            with open(self.inpaint_regions_file) as file:
                for idx, line in enumerate(file):
                    try:
                        self.inpaint_regions[idx] = [(float(x), float(y)) for x, y in line]
                    except ValueError:
                        continue

    def __getitem__(self, index):
        DIV_path, Vx_path, Vy_path = self.A_paths[index]

        def format_correct(file_path):
            with open(file_path) as file:
                line = file.readline()
                file.seek(0)

                return len(line.split()) == 1

        while not format_correct(DIV_path):
            self.A_paths.pop(index)
            # A_path = self.A_paths[index]
            DIV_path, Vx_path, Vy_path = self.A_paths[index]

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
        # A = np.empty([cols,rows], dtype=np.float32)
        # A = np.empty([cols,rows], dtype=np.float32)
        # A = np.empty([cols,rows], dtype=np.float32)
            # r=0
            # c=0
        def read_geo_file(path):
            with open(path) as file:
                data = list(map(float, file.read().split()))

                if len(data) > rows*cols*depth:
                    assert(len(data) == rows*cols*3)

                    data = [data[i] for i in range(2, len(data), 3)]

                return np.array(data).reshape((cols, rows), order='F')

        A_DIV = read_geo_file(DIV_path)
        A_Vx = read_geo_file(Vx_path)
        A_Vy = read_geo_file(Vy_path)

        # A = Image.fromarray(A.astype(np.uint8))
        A_DIV = resize(A_DIV, (self.opt.fineSize * 2, self.opt.fineSize), mode='constant')
        A_Vx = resize(A_Vx, (self.opt.fineSize * 2, self.opt.fineSize), mode='constant')
        A_Vy = resize(A_Vy, (self.opt.fineSize * 2, self.opt.fineSize), mode='constant')

        # A = A_DIV

        # First array is gradient in rows
        grad_y = np.gradient(A_Vy, axis=0)
        # Second array is gradient in columns
        grad_x = np.gradient(A_Vx, axis=1)

        A_DIV = grad_x + grad_y

        # plt.subplot(121)
        # io.imshow(A)
        # plt.subplot(122)
        # io.imshow(test_DIV)
        # plt.show()

        # w = A.shape[1]
        # h = A.shape[0]
        
        # w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
        # h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))

        # A = A[h_offset:h_offset + self.opt.fineSize,
        #        w_offset:w_offset + self.opt.fineSize]
        
        def threshold(image):
            def threshold_pixel(pixel):
                if pixel.shape:
                    return np.array(list(map(threshold_pixel, pixel)))

                if pixel < -self.opt.div_threshold:
                    return -1
                elif pixel > self.opt.div_threshold:
                    return 1
                else:
                    return 0
            
            return np.array(list(map(threshold_pixel, image)))

        def skeleton(image):
            pos = image > 0
            neg = image < 0 

            pos_skel = skeletonize(pos)
            neg_skel = skeletonize(neg)

            return 0.0 + pos_skel - neg_skel


        # def remove_small_components(image):
        #     conn_comp = label((image*2).astype(int), neighbors=8, background=1)

        #     areas = [prop.area for prop in regionprops(conn_comp, image)]

        #     perc_70th = sorted(areas)[int(.7 * len(areas))]
        #     remove_small_objects(conn_comp, perc_70th-1, connectivity=2, in_place=True)

        #     image[np.where(np.invert(conn_comp.astype(bool)))] = 0.5

        #     return image

        # B = np.zeros(A.shape)
        # if self.process.startswith("threshold"):
        # elif self.process.startswith("skeleton"):
        #     B = skeleton(A)

        if not self.inpaint_regions[index]:
            with open(self.inpaint_regions_file, 'w') as file:
                # file_inpaint_regions = [(float(x), float(y)) for line in file.read().split() for x, y in line]
                # file.seek(0)

                w = A_DIV.shape[1]
                h = A_DIV.shape[0]

                w_offset = random.randint(0, max(0, w - 100 - 1))
                h_offset = random.randint(0, max(0, h - 100 - 1))

                self.inpaint_regions[index] = (w_offset, h_offset)
                # file_inpaint_regions[index] = (w_offset, h_offset)

                for region in self.inpaint_regions:
                    if not region:
                        file.write("0, 0")
                    else:
                        file.write("{0}, {1}".format(w_offset, h_offset))
        
        w_offset, h_offset = self.inpaint_regions[index]
        # print(w_offset, h_offset)

        # io.imshow(B)
        # io.show()


        # print(A.shape)
        # print(B.shape)

        B_DIV = A_DIV.copy()
        B_DIV[h_offset:h_offset+100, w_offset:w_offset+100] = 0
        
        B_Vx = A_Vx.copy()
        B_Vx[h_offset:h_offset+100, w_offset:w_offset+100] = 0
        
        B_Vy = A_Vy.copy()
        B_Vy[h_offset:h_offset+100, w_offset:w_offset+100] = 0
        
        mask = np.zeros(B_DIV.shape, dtype=np.uint8)
        mask[h_offset:h_offset+100, w_offset:w_offset+100] = 1
        mask = np.expand_dims(mask, 2)
        mask = torch.LongTensor(mask.transpose(2, 0, 1))
        

        def create_one_hot(image):
            ridge = skeletonize(image >= self.opt.div_threshold).astype(float)
            subduction = skeletonize(image <= -self.opt.div_threshold).astype(float)
            plate = np.ones(ridge.shape, dtype=float)
            plate[np.where(np.logical_or(ridge == 1, subduction == 1))] = 0

            return np.stack((ridge, plate, subduction), axis=2)

        A = create_one_hot(A_DIV)
        B = A.copy()
        B[h_offset:h_offset+100, w_offset:w_offset+100] = [0, 1, 0]

        A_DIV = np.interp(A_DIV, [np.min(A_DIV), np.max(A_DIV)], [-1, 1])
        A_Vx = np.interp(A_Vx, [np.min(A_Vx), np.max(A_Vx)], [-1, 1])
        A_Vy = np.interp(A_Vy, [np.min(A_Vy), np.max(A_Vy)], [-1, 1])

        B_DIV = np.interp(B_DIV, [np.min(B_DIV), np.max(B_DIV)], [-1, 1])
        B_Vx = np.interp(B_Vx, [np.min(B_Vx), np.max(B_Vx)], [-1, 1])
        B_Vy = np.interp(B_Vy, [np.min(B_Vy), np.max(B_Vy)], [-1, 1])
        # B = create_one_hot(B_DIV)

        # if self.process.startswith("skeleton"):
        #     A = skeleton(A)
        #     B = skeleton(B)


        def process_image(A, B, discrete=False):
            # if len(A.shape) < 3:
            #     A = np.tile(A, (3, 1, 1)).transpose(1, 2, 0)
            #     B = np.tile(B, (3, 1, 1)).transpose(1, 2, 0)
            
            # A = A.transpose(1, 2)
            # B = B.transpose(1, 2, 0)

            

            if not discrete:
                A = np.expand_dims(A, 2)
                B = np.expand_dims(B, 2)
                A = transforms.ToTensor()(A)
                B = transforms.ToTensor()(B)

                # A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
                # B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)
            else:
                A = torch.FloatTensor(A.transpose(2, 0, 1))
                B = torch.FloatTensor(B.transpose(2, 0, 1))

            return A, B


        A, B = process_image(A, B, discrete=True)
        A_DIV, B_DIV = process_image(A_DIV, B_DIV)
        A_Vx, B_Vx = process_image(A_Vx, B_Vx)
        A_Vy, B_Vy = process_image(A_Vy, B_Vy)
        # A_DIV, B_DIV = torch.LongTensor(A_DIV.numpy()), torch.LongTensor(B_DIV.numpy())

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            A_DIV = A_DIV.index_select(2, idx)
            B_DIV = B_DIV.index_select(2, idx)
            A_Vx = A_Vx.index_select(2, idx)
            B_Vx = B_Vx.index_select(2, idx)
            A_Vy = A_Vy.index_select(2, idx)
            B_Vy = B_Vy.index_select(2, idx)
            mask = mask.index_select(2, idx)


        return {'A': A, 'B': B,
                'A_DIV': A_DIV, 'B_DIV': B_DIV,
                'A_Vx': A_Vx, 'B_Vx': B_Vx,
                'A_Vy': A_Vy, 'B_Vy': B_Vy,
                'mask': mask,
                'A_paths': DIV_path, 'B_paths': os.path.splitext(DIV_path)[0] + '_out' + os.path.splitext(DIV_path)[1]}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'GeoDataset'


# if __name__ == '__main__':
# import matplotlib.pyplot as plt
# import skimage.io as io

# os.chdir('..')
# options_dict = dict(dataroot='geology/data', phase='test', inpaint_file_dir='geology/data', process="skeleton", resize_or_crop='resize_and_crop',
#     loadSize=256, fineSize=256, which_direction='AtoB', input_nc=1, output_nc=1, no_flip=False, div_threshold=0.003)
# Options = namedtuple('Options', options_dict.keys())
# opt = Options(*options_dict.values())
# geo = GeoDataset()
# geo.initialize(opt)

# stuff = geo[0]

# dataloader = torch.utils.data.DataLoader(
#     geo,
#     batch_size=1,
#     shuffle=True,
#     num_workers=1)

# for i, batch in enumerate(dataloader):
#     print(i)

# print(np.max(stuff['A'].numpy()), np.min(stuff['A'].numpy()))
# print(np.max(stuff['B'].numpy()), np.min(stuff['B'].numpy()))
# plt.subplot(141)
# io.imshow(stuff['A'][0, :, :].numpy())
# plt.subplot(142)
# io.imshow(stuff['A'][1, :, :].numpy())
# plt.subplot(143)
# io.imshow(stuff['A'][2, :, :].numpy())
# plt.subplot(144)
# io.imshow(stuff['A'].numpy().transpose(1, 2, 0))
# plt.show()

# plt.subplot(131)
# io.imshow(stuff['A_cont'].numpy().squeeze())
# plt.subplot(132)
# io.imshow(stuff['B_cont'].numpy().squeeze())
# plt.subplot(133)
# io.imshow(stuff['mask'].numpy().transpose(1, 2, 0).squeeze())
# io.show()

# plt.subplot(141)
# io.imshow(stuff['B'][0, :, :].numpy())
# plt.subplot(142)
# io.imshow(stuff['B'][1, :, :].numpy())
# plt.subplot(143)
# io.imshow(stuff['B'][2, :, :].numpy())
# plt.subplot(144)
# io.imshow(stuff['B'].numpy().transpose(1, 2, 0))
# plt.show()

# plt.subplot(231)
# io.imshow(stuff['A'].numpy().squeeze().transpose(1, 2, 0))
# plt.subplot(232)
# io.imshow(stuff['B'].numpy().squeeze().transpose(1, 2, 0))
# plt.subplot(233)
# io.imshow(stuff['mask'].numpy().transpose(1, 2, 0).squeeze())
# io.imshow(stuff['A_DIV'].numpy().squeeze())
# plt.subplot(234)
# io.imshow(stuff['B_DIV'].numpy().squeeze())
# plt.subplot(425)
# io.imshow(stuff['A_Vx'].numpy().squeeze())
# plt.subplot(426)
# io.imshow(stuff['B_Vx'].numpy().squeeze())
# plt.subplot(427)
# io.imshow(stuff['A_Vy'].numpy().squeeze())
# plt.subplot(428)
# io.imshow(stuff['B_Vy'].numpy().squeeze())
# plt.show()
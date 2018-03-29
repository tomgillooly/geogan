import glob
import os.path
import random
import re
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

EARTH_SURFACE_VELOCITY = 3.956e-2
MODEL_SURFACE_VELOCITY = 700

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

        DIV_paths = sorted(DIV_paths)
        Vx_paths = sorted(Vx_paths)
        Vy_paths = sorted(Vy_paths)

        # self.A_paths = sorted(self.A_paths)

        self.A_paths = list(zip(DIV_paths, Vx_paths, Vy_paths))
        
        print(len(self(DIV_paths)))

        assert(len(self.A_paths) > 0)

        assert(opt.resize_or_crop == 'resize_and_crop')

        self.inpaint_regions_file = os.path.join(opt.inpaint_file_dir, opt.phase, 'inpaint_regions')
        self.inpaint_regions = [None]*len(self.A_paths)
        # self.inpaint_regions_file = os.path.join(opt.phase, 'inpaint_regions')
        
        if os.path.exists(self.inpaint_regions_file):
            with open(self.inpaint_regions_file) as file:
                for idx, line in enumerate(file):
                    try:
                        self.inpaint_regions[idx] = tuple([int(param.lstrip()) for param in line.rstrip().split(',')])
                    except ValueError:
                        continue

    def __getitem__(self, index):
        DIV_path, Vx_path, Vy_path = self.A_paths[index]

        series_number = re.search('serie(\d+)', DIV_path).group(1)

        # Check that they're all the same series number, as glob doesn't always
        # return the files in the correct order
        assert(series_number in Vx_path)
        assert(series_number in Vy_path)

        def format_correct(file_path):
            with open(file_path) as file:
                line = file.readline()
                file.seek(0)

                return len(line.split()) == 1

        while not format_correct(DIV_path):
            self.A_paths.pop(index)
            # A_path = self.A_paths[index]
            DIV_path, Vx_path, Vy_path = self.A_paths[index]

        series_number = re.search('serie(\d+)', DIV_path).group(1)

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

        # It is possible to do an interpolation here, but it's really slow
        # and ends up looking about the same
        def read_geo_file(path):
            with open(path) as file:
                data = list(map(float, file.read().split()))

                if len(data) > rows*cols*depth:
                    assert(len(data) == rows*cols*3)

                    x = np.array([data[i] for i in range(0, len(data), 3)])
                    y = np.array([data[i] for i in range(1, len(data), 3)])
                    data = np.array([data[i] for i in range(2, len(data), 3)]).reshape((rows, cols), order='C')

                    return x, y, data

                return np.array(data).reshape((rows, cols), order='C')

        A_DIV_orig = read_geo_file(DIV_path)
        x, y, A_Vx = read_geo_file(Vx_path)
        x, y, A_Vy = read_geo_file(Vy_path)

        A_Vx *= MODEL_SURFACE_VELOCITY / EARTH_SURFACE_VELOCITY
        A_Vy *= MODEL_SURFACE_VELOCITY / EARTH_SURFACE_VELOCITY

        # print(A_Vx.ravel().max())
        # print(A_Vy.ravel().max())

        # A = Image.fromarray(A.astype(np.uint8))
        A_DIV_orig = resize(A_DIV_orig, (self.opt.fineSize, self.opt.fineSize * 2), mode='constant')
        A_Vx = resize(A_Vx, (self.opt.fineSize, self.opt.fineSize * 2), mode='constant')
        A_Vy = resize(A_Vy, (self.opt.fineSize, self.opt.fineSize * 2), mode='constant')

        A_DIV = A_DIV_orig

        # grad_y = np.flip(np.gradient(np.flip(A_Vy, axis=0), np.unique(y), axis=0, edge_order=2), axis=0)
        grad_y = np.flip(np.gradient(np.flip(A_Vy, axis=0), 1e-2, axis=0, edge_order=2), axis=0)
        # grad_x = np.gradient(A_Vx, np.unique(x), axis=1, edge_order=2)
        grad_x = np.gradient(A_Vx, 1e-2, axis=1, edge_order=2)

        # A_DIV = grad_x + grad_y

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

        def create_one_hot(image):
            ridge = skeletonize(image >= self.opt.div_threshold).astype(float)
            subduction = skeletonize(image <= -self.opt.div_threshold).astype(float)
            plate = np.ones(ridge.shape, dtype=float)
            plate[np.where(np.logical_or(ridge == 1, subduction == 1))] = 0

            return np.stack((ridge, plate, subduction), axis=2)

        A = create_one_hot(A_DIV)

        if not self.inpaint_regions[index]:
            # If we have a batch size > 1, we need to update our inpaint regions
            # from the other thread
            # Also, we have to do this in stages because the file.seek(0) doesnt seem
            # to work when we do it twice inside one context manager block
            with open(self.inpaint_regions_file, 'a+') as file:
                file.seek(0)

                for idx, line in enumerate(file):

                    try:
                        self.inpaint_regions[idx] = tuple([int(param.lstrip()) for param in line.rstrip().split(',')])
                    except ValueError:
                        continue


            w = A_DIV.shape[1]
            h = A_DIV.shape[0]

            success = False

            while not success:

                w_offset = random.randint(0, max(0, w - 100 - 1))
                h_offset = random.randint(0, max(0, h - 100 - 1))

                layer = int(round(random.random())*2)

                if np.sum(A[h_offset:h_offset+100, w_offset:w_offset+100, layer]) > 0 and np.sum(A[h_offset:h_offset+100, w_offset:w_offset+100, 2-layer]) > 0:
                    self.inpaint_regions[index] = (w_offset, h_offset, layer)
                    
                    success = True


            with open(self.inpaint_regions_file, 'w') as file:
                file.seek(0)
                for idx, region in enumerate(self.inpaint_regions):
                    if not region:
                        file.write("None\n")
                    else:
                        file.write("{0},{1},{2}\n".format(*region))

                file.truncate()
        
        w_offset, h_offset, layer = self.inpaint_regions[index]
        # print(w_offset, h_offset, layer)

        mask_x1 = w_offset
        mask_x2 = w_offset+100
        
        mask_y1 = h_offset
        mask_y2 = h_offset+100

        B_DIV = A_DIV.copy()
        B_DIV[mask_y1:mask_y2, mask_x1:mask_x2] = 0
        
        B_Vx = A_Vx.copy()
        B_Vx[mask_y1:mask_y2, mask_x1:mask_x2] = 0
        
        B_Vy = A_Vy.copy()
        B_Vy[mask_y1:mask_y2, mask_x1:mask_x2] = 0
        
        mask = np.zeros(B_DIV.shape, dtype=np.uint8)
        mask[mask_y1:mask_y2, mask_x1:mask_x2] = 1
        
        B = A.copy()

        B[:, :, 1][np.where(np.logical_and(mask, B[:, :, layer]))] = 1
        B[mask_y1:mask_y2, mask_x1:mask_x2, layer] = 0
            
        mask = np.expand_dims(mask, 2)
        mask = torch.ByteTensor(mask.transpose(2, 0, 1))

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

            tmp = mask_x1
            mask_x1 = mask.shape[2] - mask_x2
            mask_x2 = mask.shape[2] - tmp

        mask_x1 = torch.LongTensor([mask_x1])
        mask_x2 = torch.LongTensor([mask_x2])
        mask_y1 = torch.LongTensor([mask_y1])
        mask_y2 = torch.LongTensor([mask_y2])

        return {'A': A, 'B': B,
                'A_DIV': A_DIV, 'B_DIV': B_DIV,
                'A_Vx': A_Vx, 'B_Vx': B_Vx,
                'A_Vy': A_Vy, 'B_Vy': B_Vy,
                'mask': mask,
                'mask_x1': mask_x1, 'mask_x2': mask_x2,
                'mask_y1': mask_y1, 'mask_y2': mask_y2,
                'A_paths': DIV_path,
                'B_paths': os.path.splitext(DIV_path)[0] + '_out' + os.path.splitext(DIV_path)[1]
                }

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'GeoDataset'


# if __name__ == '__main__':
# import matplotlib.pyplot as plt
# import skimage.io as io
# from skimage.filters import roberts

# os.chdir('..')
# options_dict = dict(dataroot=os.path.expanduser('~/data/geology/'), phase='test', inpaint_file_dir=os.path.expanduser('~/data/geology/'), process="skeleton", resize_or_crop='resize_and_crop',
#     loadSize=256, fineSize=256, which_direction='AtoB', input_nc=1, output_nc=1, no_flip=True, div_threshold=1000)
# Options = namedtuple('Options', options_dict.keys())
# opt = Options(*options_dict.values())
# geo = GeoDataset()
# geo.initialize(opt)

# # stuff = geo[0]


# dataloader = torch.utils.data.DataLoader(
#     geo,
#     batch_size=2,
#     shuffle=False,
#     num_workers=1)

# # batch = next(iter(dataloader))


# for i, batch in enumerate(dataloader):
#     mask = batch['mask']
#     print(batch['mask_x1'])
#     print(batch['mask_x2'])
#     print(batch['mask_y1'])
#     print(batch['mask_y2'])
#     print(mask.shape)
#     print(mask.repeat(1, 3, 1, 1).shape)
#     mask_regions = batch['B'].masked_select(mask.repeat(1, 3, 1, 1)).view(2, 3, 100, 100)

#     _, axes = plt.subplots(2, 2)
#     ax = axes.ravel()
#     ax[0].imshow(batch['B'][0, :, :, :].numpy().squeeze().transpose(1, 2, 0))
#     ax[1].imshow(batch['B'][1, :, :, :].numpy().squeeze().transpose(1, 2, 0))
#     ax[2].imshow(mask_regions[0, :, :, :].numpy().squeeze().transpose(1, 2, 0))
#     ax[3].imshow(mask_regions[1, :, :, :].numpy().squeeze().transpose(1, 2, 0))

#     plt.show()

#     break
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

# plt.subplot(421)
# io.imshow(stuff['A'].numpy().squeeze().transpose(1, 2, 0))
# plt.subplot(422)
# io.imshow(stuff['B'].numpy().squeeze().transpose(1, 2, 0))
# plt.subplot(423)
# # io.imshow(stuff['mask'].numpy().transpose(1, 2, 0).squeeze())
# io.imshow(stuff['A_DIV'].numpy().squeeze())
# plt.subplot(424)
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
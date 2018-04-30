import glob
import os
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

class DataGenException(Exception):
    def __init__(self, error_msg, img):
        super().__init__(error_msg)
        self.img = img


def get_series_number(path):
    match = re.search('serie1?_?(\d+)_?', path)            

    return int(match.group(1))


def read_geo_file(path):
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


def get_file_tag(path):
    return os.path.splitext(path)[0].split('_')[-1]


def create_one_hot(image, threshold):
    ridge = skeletonize(image >= threshold).astype(float)
    subduction = skeletonize(image <= -threshold).astype(float)
    plate = np.ones(ridge.shape, dtype=float)
    plate[np.where(np.logical_or(ridge == 1, subduction == 1))] = 0

    return np.stack((ridge, plate, subduction), axis=2)


def mask_out_inpaint_region(im, mask):
    im = im.copy()
    im[np.where(mask)] = 0

    return im


class GeoDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot, opt.phase)
        self.A_paths = []

        data_paths = self.get_dat_files(self.dir_A)

        # If there's no continent data, remove so we don't end up with zero paths after zip
        self.A_paths = list(zip(*[path for path in data_paths if path]))
        
        assert(len(self.A_paths) > 0)

        assert(opt.resize_or_crop == 'resize_and_crop')

        self.inpaint_regions_file = os.path.join(opt.inpaint_file_dir, opt.phase, 'inpaint_regions')
        self.inpaint_regions = [None]*len(self.A_paths)
        
        if os.path.exists(self.inpaint_regions_file):
            with open(self.inpaint_regions_file) as file:
                for idx, line in enumerate(file):
                    try:
                        self.inpaint_regions[idx] = tuple([int(param.lstrip()) for param in line.rstrip().split(',')])
                    except ValueError:
                        continue


    def get_dat_files(self, topdir):
        # DIV_paths = glob.glob(os.path.join(topdir, '*_DIV.dat'))
        # Vx_paths = glob.glob(os.path.join(topdir, '*_Vx.dat'))
        # Vy_paths = glob.glob(os.path.join(topdir, '*_Vy.dat'))

        # DIV_paths = sorted(DIV_paths)
        # Vx_paths = sorted(Vx_paths)
        # Vy_paths = sorted(Vy_paths)

        DIV_paths = []
        Vx_paths = []
        Vy_paths = []
        cont_paths = []

        self.opt.num_folders = 0

        self.folder_id_lookup = {}

        # Remove trailing slashm, if there is one (sometimes there's not)
        topdir = topdir.rstrip('/')
        for root, dirs, _ in os.walk(topdir):
            # Re-add one to length for the trailing slash
            self.folder_id_lookup[root[len(topdir)+1:]] = self.opt.num_folders

            self.opt.num_folders += 1

            DIV_paths += sorted(glob.glob(os.path.join(root, '*_DIV.dat')), key=get_series_number)
            Vx_paths += sorted(glob.glob(os.path.join(root, '*_Vx.dat')), key=get_series_number)
            Vy_paths += sorted(glob.glob(os.path.join(root, '*_Vy.dat')), key=get_series_number)
            cont_paths += sorted(glob.glob(os.path.join(root, '*_cont.dat')), key=get_series_number)

            # Make sure they're in numerical order before we recurse into them
            dirs.sort(key=int)

            # for directory in dirs:
            #     DIV_paths += glob.glob(os.path.join(root, directory, '*_DIV.dat'))
            #     Vx_paths += glob.glob(os.path.join(root, directory, '*_Vx.dat'))
            #     Vy_paths += glob.glob(os.path.join(root, directory, '*_Vy.dat'))

        
        return DIV_paths, Vx_paths, Vy_paths, cont_paths


    def update_inpaint_regions_from_file(self):
        with open(self.inpaint_regions_file, 'a+') as file:
            file.seek(0)

            for idx, line in enumerate(file):
                try:
                    self.inpaint_regions[idx] = tuple([int(param.lstrip()) for param in line.rstrip().split(',')])

                    assert(len(self.inpaint_regions[idx]) == 3)
                except ValueError:
                    continue


    def update_inpaint_regions_file(self):
        with open(self.inpaint_regions_file, 'w') as file:
            file.seek(0)
            for idx, region in enumerate(self.inpaint_regions):
                if not region:
                    file.write("None\n")
                else:
                    file.write("{0},{1},{2}\n".format(*region))

            file.truncate()


    def get_inpaint_region(self, index, A, h, w):
        if not self.inpaint_regions[index]:
            # If we have a batch size > 1, we need to update our inpaint regions
            # from the other thread
            # Also, we have to do this in stages because the file.seek(0) doesnt seem
            # to work when we do it twice inside one context manager block
            # We can do this with multiprocessing Manager lists, but that can be done later
            self.update_inpaint_regions_from_file()

            # success = False

            # while not success:
            # 100 tries to find mask region
            for i in range(100):

                w_offset = random.randint(0, max(0, w - 100 - 1))
                h_offset = random.randint(0, max(0, h - 100 - 1))

                layer = int(round(random.random())*2)

                if np.sum(A[h_offset:h_offset+100, w_offset:w_offset+100, layer]) > 0 and np.sum(A[h_offset:h_offset+100, w_offset:w_offset+100, 2-layer]) > 0:
                    self.inpaint_regions[index] = (w_offset, h_offset, layer)

                    break

            if i == 99:
                raise DataGenException("Couldn't choose mask region in file " + self.A_paths[index][0], A)

            self.update_inpaint_regions_file()

        return self.inpaint_regions[index]


    def __getitem__(self, index):
        A_paths = self.A_paths[index]

        match = re.search('(/\d+)?/serie(\d+)', A_paths[0])

        folder_id = self.folder_id_lookup[match.group(1)[1:] if match.group(1) else '']

        dir_tag = '_' + match.group(1)[1:] + '_' if match.group(1) else '_'
        series_number = int(match.group(2))

        # Check that all files are of the same series number, as glob doesn't always
        # return the files in the correct order
        
        s_no = series_number - 100000

        assert all([get_series_number(path) == s_no for path in A_paths[1:]]), A_paths

        series = 'serie' + dir_tag + str(series_number)

        data = OrderedDict([(get_file_tag(path), read_geo_file(path)) for path in A_paths])

        rows = len(np.unique(data['Vx']['y']))
        cols = len(np.unique(data['Vx']['x']))

        # It is possible to do an interpolation here, but it's really slow
        # and ends up looking about the same
        for key in data.keys():
            data[key]['values'] = data[key]['values'].reshape((rows, cols), order='C')
            data[key]['values'] = resize(data[key]['values'], (self.opt.fineSize, self.opt.fineSize * 2), mode='constant')

        rows = 256
        cols = 512

        # A_DIV = data['DIV']['values']
        # A_Vx = data['Vx']['values']
        # A_Vy = data['Vy']['values']

        # Create discrete image before we normalise
        A = create_one_hot(data['DIV']['values'], self.opt.div_threshold)
        
        # We're done with x/y data now, so discard
        A_data = [data[key]['values'] for key in data.keys() if key != 'cont']
        # Normalise
        A_DIV, A_Vx, A_Vy = A_data

        folder_dir = os.path.dirname(A_paths[0])

        def get_norm_data(tag):
            with open(os.path.join(folder_dir, tag + '_norm.dat')) as file:
                dmin, dmax = [float(x) for x in file.read().split()]

                return dmin, dmax


        A_DIV = np.interp(A_DIV, get_norm_data('DIV'), [-1, 1])
        A_Vx = np.interp(A_Vx, get_norm_data('Vx'), [-1, 1])
        A_Vy = np.interp(A_Vy, get_norm_data('Vy'), [-1, 1])

        # if self.opt.continent_data

        # Don't need this, as we're just normalising anyway
        # A_Vx *= MODEL_SURFACE_VELOCITY / EARTH_SURFACE_VELOCITY
        # A_Vy *= MODEL_SURFACE_VELOCITY / EARTH_SURFACE_VELOCITY
        
        w_offset, h_offset, layer = self.get_inpaint_region(index, A, rows, cols)
        # w_offset, h_offset, layer = self.inpaint_regions[index]
        # print(w_offset, h_offset, layer)

        mask_x1 = w_offset
        mask_x2 = w_offset+100
        
        mask_y1 = h_offset
        mask_y2 = h_offset+100

        mask = np.zeros((rows, cols), dtype=np.uint8)
        mask[mask_y1:mask_y2, mask_x1:mask_x2] = 1

        # B_DIV = A_DIV.copy()
        # B_DIV[mask_y1:mask_y2, mask_x1:mask_x2] = 0
        
        # B_Vx = A_Vx.copy()
        # B_Vx[mask_y1:mask_y2, mask_x1:mask_x2] = 0
        
        # B_Vy = A_Vy.copy()
        # B_Vy[mask_y1:mask_y2, mask_x1:mask_x2] = 0

        B_data = [mask_out_inpaint_region(data, mask) for data in [A_DIV, A_Vx, A_Vy]]
        
        B = A.copy()

        if self.opt.inpaint_single_class:
            B[:, :, 1][np.where(np.logical_and(mask, B[:, :, layer]))] = 1
            B[mask_y1:mask_y2, mask_x1:mask_x2, layer] = 0
        else:
            B[np.where(mask)] = [0, 1, 0]
            
        mask = np.expand_dims(mask, 2)
        mask = torch.ByteTensor(mask.transpose(2, 0, 1)).clone()

        # A_DIV = np.interp(A_DIV, [np.min(A_DIV), np.max(A_DIV)], [-1, 1])
        # A_Vx = np.interp(A_Vx, [np.min(A_Vx), np.max(A_Vx)], [-1, 1])
        # A_Vy = np.interp(A_Vy, [np.min(A_Vy), np.max(A_Vy)], [-1, 1])

        # B_DIV = np.interp(B_DIV, [np.min(B_DIV), np.max(B_DIV)], [-1, 1])
        # B_Vx = np.interp(B_Vx, [np.min(B_Vx), np.max(B_Vx)], [-1, 1])
        # B_Vy = np.interp(B_Vy, [np.min(B_Vy), np.max(B_Vy)], [-1, 1])


        def process_image(A, B, discrete=False):
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

        B_DIV, B_Vx, B_Vy = B_data

        if self.opt.continent_data:
            if 'cont' in data.keys():
                continents = data['cont']['values']
            else:
                continents = np.zeros((rows, cols))

        A, B = process_image(A, B, discrete=True)
        A_DIV, B_DIV = process_image(A_DIV, B_DIV)
        A_Vx, B_Vx = process_image(A_Vx, B_Vx)
        A_Vy, B_Vy = process_image(A_Vy, B_Vy)

        if self.opt.continent_data:
            continents = (continents > 0).astype(np.uint8)
            continents = np.expand_dims(continents, 2)
            continents = continents.transpose(2, 0, 1)
            
            continents = torch.ByteTensor(continents).clone()

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

            if self.opt.continent_data:
                continents = continents.index_select(2, idx)

            mask = mask.index_select(2, idx)

            tmp = mask_x1
            mask_x1 = mask.shape[2] - mask_x2
            mask_x2 = mask.shape[2] - tmp

        mask_x1 = torch.LongTensor([mask_x1])
        mask_x2 = torch.LongTensor([mask_x2])
        mask_y1 = torch.LongTensor([mask_y1])
        mask_y2 = torch.LongTensor([mask_y2])

        data =  {'A': A, 'B': B,
                'A_DIV': A_DIV, 'B_DIV': B_DIV,
                'A_Vx': A_Vx, 'B_Vx': B_Vx,
                'A_Vy': A_Vy, 'B_Vy': B_Vy,
                'mask': mask,
                'mask_x1': mask_x1, 'mask_x2': mask_x2,
                'mask_y1': mask_y1, 'mask_y2': mask_y2,
                'A_paths': os.path.join(self.dir_A, series),
                'B_paths': os.path.join(self.dir_A, series + '_inpainted'),
                'series_number': int(dir_tag[1:-1] + str(series_number)),
                'folder_id': folder_id
                }

        if self.opt.continent_data:
            data['continents'] = continents

        return data

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'GeoDataset'


# if __name__ == '__main__':
# import matplotlib.pyplot as plt
# import skimage.io as io
# from skimage.filters import roberts

# # os.chdir('..')
# # options_dict = dict(dataroot=os.path.expanduser('~/data/geology/'), phase='test', inpaint_file_dir=os.path.expanduser('~/data/geology/'), process="skeleton", resize_or_crop='resize_and_crop',
# options_dict = dict(dataroot='test_data/with_continents', phase='', inpaint_file_dir='test_data/with_continents', process="skeleton", resize_or_crop='resize_and_crop',
#     loadSize=256, fineSize=256, which_direction='AtoB', input_nc=1, output_nc=1, no_flip=True, div_threshold=1000, inpaint_single_class=False, continent_data=True)
# Options = namedtuple('Options', options_dict.keys())
# opt = Options(*options_dict.values())
# geo = GeoDataset()
# geo.initialize(opt)

# stuff = geo[0]

# fig, ax = plt.subplots(5, 2)
# ax = ax.ravel()
# ax[0].imshow(stuff['A'].numpy().squeeze().transpose(1, 2, 0))
# ax[1].imshow(stuff['A_DIV'].numpy().squeeze())
# ax[2].imshow(stuff['A_Vx'].numpy().squeeze())
# ax[3].imshow(stuff['A_Vy'].numpy().squeeze())
# ax[4].imshow(stuff['B'].numpy().squeeze().transpose(1, 2, 0))
# ax[5].imshow(stuff['B_DIV'].numpy().squeeze())
# ax[6].imshow(stuff['B_Vx'].numpy().squeeze())
# ax[7].imshow(stuff['B_Vy'].numpy().squeeze())
# ax[8].imshow(stuff['mask'].numpy().squeeze())
# plt.show()


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
# io.imshow(stuff['continents'].numpy().squeeze())
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
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

        for root, dirs, files in os.walk(self.dir_A):
            self.A_paths += [os.path.join(root, file) for file in files]

        self.A_paths = [path for path in self.A_paths if path.endswith(".dat")]

        assert(len(self.A_paths) > 0)
        self.A_paths = sorted(self.A_paths)

        assert(opt.resize_or_crop == 'resize_and_crop')

        self.inpaint_regions_file = os.path.join(opt.dataroot, opt.phase, 'inpaint_regions')
        
        if os.path.exists(self.inpaint_regions_file):
            with open(self.inpaint_regions_file) as file:
                self.inpaint_regions = [(float(x), float(y)) for line in file.read().split() for x, y in line]
        else:
            self.inpaint_regions = [None]*len(self.A_paths)

    def __getitem__(self, index):
        A_path = self.A_paths[index]

        def format_correct(file_path):
            with open(file_path) as file:
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
        A = resize(A, (self.opt.fineSize * 2, self.opt.fineSize), mode='constant')

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

                if pixel < -1000:
                    return 0
                elif pixel > 1000:
                    return 1
                else:
                    return 0.5
            
            return np.array(list(map(threshold_pixel, A)))

        # def skeleton(image):
        #     pos = image > 1000
        #     neg = image < -1000

        #     pos_skel = skeletonize(pos)
        #     neg_skel = skeletonize(neg)

        #     return 0.5 + pos_skel*0.5 - neg_skel*0.5


        # def remove_small_components(image):
        #     conn_comp = label((image*2).astype(int), neighbors=8, background=1)

        #     areas = [prop.area for prop in regionprops(conn_comp, image)]

        #     perc_70th = sorted(areas)[int(.7 * len(areas))]
        #     remove_small_objects(conn_comp, perc_70th-1, connectivity=2, in_place=True)

        #     image[np.where(np.invert(conn_comp.astype(bool)))] = 0.5

        #     return image

        # B = np.zeros(A.shape)
        # if self.process.startswith("threshold"):
        B = threshold(A)
        # elif self.process.startswith("skeleton"):
        #     B = skeleton(A)

        # if "remove_small_components" in self.process:
        #     B = remove_small_components(B)

        if not self.inpaint_regions[index]:
            with open(self.inpaint_regions_file, 'w') as file:
                # file_inpaint_regions = [(float(x), float(y)) for line in file.read().split() for x, y in line]
                # file.seek(0)

                w = A.shape[1]
                h = A.shape[0]

                w_offset = random.randint(0, max(0, w - 20 - 1))
                h_offset = random.randint(0, max(0, h - 20 - 1))

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

        B[h_offset:h_offset+100, w_offset:w_offset+100] = 0.5
        # io.imshow(B)
        # io.show()

        # print(A.shape)
        # print(B.shape)

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
                'A_paths': A_path, 'B_paths': os.path.splitext(A_path)[0] + '_thresh' + os.path.splitext(A_path)[1]}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'GeoDataset'


# if __name__ == '__main__':
# import matplotlib.pyplot as plt
# import skimage.io as io

# os.chdir('..')
# print(os.getcwd())
# # options_dict = dict(dataroot='/storage/Datasets/Geology-NicolasColtice/DS2-1810-RAW-DAT', phase='train', resize_or_crop='resize_and_crop', loadSize=256, fineSize=256, which_direction='AtoB', input_nc=1, output_nc=1, no_flip=True)
# options_dict = dict(dataroot='geology/data', phase='train', process="threshold_remove_small_components", resize_or_crop='resize_and_crop', loadSize=256, fineSize=256, which_direction='AtoB', input_nc=1, output_nc=1, no_flip=True)
# Options = namedtuple('Options', options_dict.keys())
# opt = Options(*options_dict.values())
# geo = GeoDataset()
# geo.initialize(opt)

# stuff = geo[1]

# # dataloader = torch.utils.data.DataLoader(
# #     geo,
# #     batch_size=1,
# #     shuffle=True,
# #     num_workers=1)

# # for i, batch in enumerate(dataloader):
# #     print(i)

# print(np.max(stuff['A'].numpy()), np.min(stuff['A'].numpy()))
# print(np.max(stuff['B'].numpy()), np.min(stuff['B'].numpy()))

# plt.subplot(121)
# io.imshow(stuff['A'].numpy().transpose(1, 2, 0).squeeze())
# plt.subplot(122)
# io.imshow(stuff['B'].numpy().transpose(1, 2, 0).squeeze())
# plt.show()
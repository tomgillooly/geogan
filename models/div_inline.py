import torch
import torchvision.transforms as transforms
import torch.autograd as autograd
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .second_head import SecondHead

from data.geo_pickler import GeoPickler

import torch.nn.functional as F
import numpy as np
import re
from functools import reduce

import skimage.io as io

from scipy.spatial.distance import directed_hausdorff, euclidean
from skimage.filters import roberts

from metrics.hausdorff import get_hausdorff, get_hausdorff_exc

import sys

# Weight init procedure taken from  https://github.com/pytorch/examples/blob/master/dcgan/main.py#L131
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('InstanceNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0.0)


def get_innermost(module, block_name=None):
    module_name_re = re.compile('(.*?)\(')
    parent_name = block_name if block_name else module_name_re.match(repr(module)).group(1)

    children_list = list(module.children())
    child_names = [module_name_re.match(repr(m)).group(1) for m in children_list]

    if parent_name in child_names:
        return get_innermost(children_list[child_names.index(parent_name)])
    elif 'Sequential' in child_names:
        return get_innermost(children_list[child_names.index('Sequential')], parent_name)
    else:
        # Just assume the innermost block is right in the middle at this stage
        return children_list[int(len(children_list)/2)]


def get_downsample(net):
    downsample = [m.stride[0] for m in net.modules() if repr(m).startswith('Conv2d')]
    downsample = reduce(lambda x, y: x*y, downsample)

    return downsample


class DiscriminatorWGANGP(torch.nn.Module):

    def __init__(self, in_dim, image_dims, dim=64):
        super(DiscriminatorWGANGP, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                # Gulrajanis code uses TensorFlow batch normalisation
                torch.nn.InstanceNorm2d(out_dim, affine=True),
                torch.nn.LeakyReLU(0.2))

        self.ls = torch.nn.Sequential(
            torch.nn.Conv2d(in_dim, dim, 5, 2, 2), torch.nn.LeakyReLU(0.2),         # (b, c, x, y) -> (b, dim, x/2, y/2)
            conv_ln_lrelu(dim, dim * 2),                                # (b, dim, x/2, y/2) -> (b, dim*2, x/4, y/4)
            conv_ln_lrelu(dim * 2, dim * 4),                            # (b, dim*2, x/4, y/4) -> (b, dim*4, x/8, y/8)
            conv_ln_lrelu(dim * 4, dim * 8),                            # (b, dim*4, x/8, y/8) -> (b, dim*8, x/16, y/16)
            torch.nn.Conv2d(dim * 8, 1, 
                (int(image_dims[0]/16 + 0.5), int(image_dims[1]/16 + 0.5)))) # (b, dim*8, x/16, y/16) -> (b, 1, 1, 1)


    def forward(self, x):
        y = self.ls(x)

        return y.view(-1)


def save_output_hook(module, input, output):
    module.output = output


class DivInlineModel(BaseModel):
    def name(self):
        return 'Pix2PixGeoModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        self.p = GeoPickler('')

        # load/define networks
        # Input channels = 3 channels for input one-hot map + mask
        input_channels = opt.input_nc + 1

        if self.opt.continent_data:
            input_channels += 1

        self.netG = networks.define_G(input_channels, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        if self.isTrain and opt.num_folders > 1 and self.opt.folder_pred:
            self.netG.inner_layer = get_innermost(self.netG, 'UnetSkipConnectionBlock')
            self.netG.inner_layer.register_forward_hook(save_output_hook)

            # Image size downsampled, times number of filters
            ds_factor = get_downsample(self.netG)
            inner_im_size = (int(opt.fineSize / ds_factor), int(2*opt.fineSize / ds_factor))
            print(inner_im_size)
            GAP = torch.nn.AvgPool2d(inner_im_size)
            self.folder_fc = torch.nn.Sequential(GAP,
                torch.nn.Linear(opt.ngf*8, opt.num_folders))

            if len(self.gpu_ids) > 0:
                self.folder_fc.cuda(self.gpu_ids[0])


        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)


        if self.isTrain:
            # define loss functions

            self.criterionL2 = torch.nn.MSELoss(size_average=False)
            self.criterionCE = torch.nn.NLLLoss2d

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            
            # Just a linear decay over the last 100 iterations, by default
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        print('-----------------------------------------------')

    def set_input(self, input):
        # This model is B to A by default
        AtoB = self.opt.which_direction == 'AtoB'

        # This is a confusing leftover of the pix2pix code
        # We process the images in the geo dataset A to B, that is
        # full image to masked out
        # So we want to switch direction, as we're trying to predict the
        # full image from the masked out
        input_A     = input['A' if AtoB else 'B']
        input_B     = input['B' if AtoB else 'A']
        input_A_DIV = input['A_DIV' if AtoB else 'B_DIV']
        input_B_DIV = input['B_DIV' if AtoB else 'A_DIV']
        mask        = input['mask']

        if self.opt.continent_data:
            continents = input['cont']
        
        if len(self.gpu_ids) > 0:
            input_A     = input_A.cuda(self.gpu_ids[0], async=True)
            input_B     = input_B.cuda(self.gpu_ids[0], async=True)
            input_A_DIV = input_A_DIV.cuda(self.gpu_ids[0], async=True)
            input_B_DIV = input_B_DIV.cuda(self.gpu_ids[0], async=True)
            mask        = mask.cuda(self.gpu_ids[0], async=True)
            
            if self.opt.continent_data:
                continents = continents.cuda(self.gpu_ids[0], async=True)
        
        self.input_A        = input_A
        self.input_B        = input_B
        self.input_A_DIV    = input_A_DIV
        self.input_B_DIV    = input_B_DIV
        self.mask           = mask

        if 'A_paths' in input.keys():
            self.A_path = input['A_paths']
        elif 'folder_id' in input.keys():
            self.A_path = ['serie_{}_{:05}'.format(input['folder_id'][0], input['series_number'][0])]
            
        if self.opt.continent_data:
            self.continent_img = continents
        

        if self.opt.isTrain and self.opt.num_folders > 1 and self.opt.folder_pred:
            self.real_folder = input['folder_id']

            if len(self.gpu_ids) > 0:
                self.real_folder = self.real_folder.cuda(self.gpu_ids[0])

        mask_x1 = input['mask_x1']
        mask_x2 = input['mask_x2']
        mask_y1 = input['mask_y1']
        mask_y2 = input['mask_y2']

        self.batch_size = input_A.shape[0]
        # Masks are always the same size (for now)
        self.mask_size_y = mask_y2[0] - mask_y1[0]
        self.mask_size_x = mask_x2[0] - mask_x1[0]

    def forward(self):
        # Thresholded, one-hot divergence map with chunk missing
        self.real_A_discrete = torch.autograd.Variable(self.input_A)
        # Complete thresholded, one-hot divergence map
        self.real_B_discrete = torch.autograd.Variable(self.input_B)

        # Continuous divergence map with chunk missing
        self.real_A_DIV = torch.autograd.Variable(self.input_A_DIV)
        
        # Complete continuous divergence map
        self.real_B_DIV = torch.autograd.Variable(self.input_B_DIV)

        # Mask of inpainted region
        self.mask = torch.autograd.Variable(self.mask)

        if self.opt.continent_data:
            self.continents = torch.autograd.Variable(self.continent_img)
        
        # Produces three channel output with class probability assignments
        # Input is one-hot image with chunk missing, conditional data is mask
        self.G_input = torch.cat((self.real_A_discrete, self.mask.float()), dim=1)

        if self.opt.continent_data:
            self.G_input = torch.cat((self.G_input, self.continents.float()), dim=1)


        self.fake_B_DIV = self.netG(self.G_input)

        # tmp_dict = {'A_DIV': self.fake_B_DIV}
        # self.p.create_one_hot(tmp_dict, self.div_thresh)
        # self.fake_B_discrete = tmp_dict['A']

        if self.opt.isTrain and self.opt.num_folders > 1 and self.opt.folder_pred:
            self.fake_folder = self.folder_fc(self.netG.inner_layer.output.view(self.batch_size, -1))
            self.fake_folder = torch.nn.functional.log_softmax(self.fake_folder, dim=1)
        

    # no backprop gradients
    def test(self):
        self.real_A_discrete = torch.autograd.Variable(self.input_A, volatile=True)
        self.real_B_discrete = torch.autograd.Variable(self.input_B, volatile=True)

        self.real_A_DIV = torch.autograd.Variable(self.input_A_DIV)
        self.real_B_DIV = torch.autograd.Variable(self.input_B_DIV)

        self.mask = torch.autograd.Variable(self.mask)
        
        if self.opt.continent_data:
            self.continents = torch.autograd.Variable(self.continent_img)

        # mask_var = Variable(self.mask.float(), volatile=True)
        self.G_input = torch.cat((self.real_A_discrete, self.mask.float()), dim=1)

        if self.opt.continent_data:
            self.G_input = torch.cat((self.G_input, self.continents.float()), dim=1)
        
        self.fake_B_DIV = self.netG(self.G_input)

        # tmp_dict = {'A_DIV': self.fake_B_DIV}
        # self.p.create_one_hot(tmp_dict, self.div_thresh)
        # self.fake_B_discrete = tmp_dict['A']


        # Work out the threshold from quantification factor
        # tmp_dict = {'A_DIV': self.fake_B_DIV.data[0].numpy().squeeze()}
        # self.p.create_one_hot(tmp_dict, 0.5)
        # self.fake_B_discrete_05 = tmp_dict['A']
        # self.p.create_one_hot(tmp_dict, 0.2)
        # self.fake_B_discrete_02 = tmp_dict['A']
        # self.p.create_one_hot(tmp_dict, 0.1)
        # self.fake_B_discrete_01 = tmp_dict['A']


    # get image paths
    def get_image_paths(self):
        return self.A_path


    def backward_G(self):
        # if we aren't taking local loss, use entire image
        loss_mask = torch.ones(self.mask.shape).byte()
        loss_mask = loss_mask.cuda() if len(self.gpu_ids) > 0 else loss_mask
        loss_mask = torch.autograd.Variable(loss_mask)

        im_dims = self.mask.shape[2:]

        if self.opt.local_loss:
            loss_mask = self.mask.byte()

            # We could maybe sum across channels 2 and 3 to get these dims, once masks are different sizes
            # im_dims = self.mask_size_y[0], self.mask_size_x[0]
            im_dims = (100, 100)
        

        total_pixels = torch.sum(torch.sum(loss_mask > 0, dim=2, keepdim=True), dim=3, keepdim=True).float()

        self.fake_B_DIV_ROI = self.fake_B_DIV.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)
        self.real_B_DIV_ROI = self.real_B_DIV.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)
        self.real_B_discrete_ROI = self.real_B_discrete.masked_select(loss_mask.repeat(1, 3, 1, 1)).view(self.batch_size, 3, *im_dims)

        num_ridge_pixels = torch.sum(torch.sum(self.real_B_discrete_ROI[:, 0, :, :].unsqueeze(1),
            dim=2, keepdim=True), dim=3, keepdim=True).float()
        num_plate_pixels = torch.sum(torch.sum(self.real_B_discrete_ROI[:, 1, :, :].unsqueeze(1),
            dim=2, keepdim=True), dim=3, keepdim=True).float()
        num_subduction_pixels = torch.sum(torch.sum(self.real_B_discrete_ROI[:, 2, :, :].unsqueeze(1),
            dim=2, keepdim=True), dim=3, keepdim=True).float()

        ridge_weight = total_pixels / num_ridge_pixels + self.opt.alpha
        plate_weight = total_pixels / num_plate_pixels + self.opt.alpha
        subduction_weight = total_pixels / num_subduction_pixels + self.opt.alpha

        ridge_weight = ridge_weight.mean(dim=0, keepdim=True)
        plate_weight = plate_weight.mean(dim=0, keepdim=True)
        subduction_weight = subduction_weight.mean(dim=0, keepdim=True)
        
        pixel_weights = torch.cat((ridge_weight, plate_weight, subduction_weight), dim=1)
        pixel_weights /= torch.sum(pixel_weights + 1e-8)

        weight_mask = torch.max(pixel_weights * self.real_B_discrete_ROI, dim=1, keepdim=True)[0]

        self.weight_mask = weight_mask

        weighted_div_predicted = self.fake_B_DIV_ROI * weight_mask
        weighted_div_target = self.real_B_DIV_ROI * weight_mask

        self.loss_G_L2_DIV = self.criterionL2(
            weighted_div_predicted,
            weighted_div_target) * self.opt.lambda_A

        self.loss_G_L2 = self.loss_G_L2_DIV
        self.loss_G = self.loss_G_L2


        if self.isTrain and self.opt.num_folders > 1 and self.opt.folder_pred:
            ce_fun = self.criterionCE()
            self.folder_pred_CE = ce_fun(self.fake_folder, self.real_folder) * self.opt.lambda_D

            self.loss_G += self.folder_pred_CE

        self.loss_G.backward()


    def optimize_parameters(self, **kwargs):
        # Doesn't do anything with discriminator, just populates input (conditional), 
        # target and generated data in object
        self.forward()

        self.optimizer_G.zero_grad()

        self.backward_G()
        
        self.optimizer_G.step()
        

    def get_current_errors(self):
        errors = [
            ('G', self.loss_G.data.item()),
            ('G_L2', self.loss_G_L2.data.item()),
            ]

        if self.isTrain and self.opt.num_folders > 1 and self.opt.folder_pred:
            errors.append(('folder_CE', self.folder_pred_CE.data[0]))

        return OrderedDict(errors)


    def get_current_visuals(self):
        # print(np.unique(self.real_A_discrete.data))
        # print(self.fake_B_discrete.data.shape)

        mask_edge = roberts(self.mask.data.cpu().numpy()[0, ...].squeeze())
        mask_edge_coords = np.where(mask_edge)

        visuals = []

        real_A_discrete = util.tensor2im(self.real_A_discrete.data)
        real_A_discrete[mask_edge_coords] = np.max(real_A_discrete)
        visuals.append(('input_one_hot', real_A_discrete))

        real_B_discrete = util.tensor2im(self.real_B_discrete.data)
        real_B_discrete[mask_edge_coords] = np.max(real_B_discrete)
        visuals.append(('ground_truth_one_hot', real_B_discrete))

        real_A_DIV = util.tensor2im(self.real_A_DIV.data)
        real_A_DIV[mask_edge_coords] = np.max(real_A_DIV)
        visuals.append(('input_divergence', real_A_DIV))

        real_B_DIV = util.tensor2im(self.real_B_DIV.data)
        real_B_DIV[mask_edge_coords] = np.max(real_B_DIV)
        visuals.append(('ground_truth_divergence', real_B_DIV))

        fake_B_DIV = util.tensor2im(self.fake_B_DIV.data)
        fake_B_DIV[mask_edge_coords] = np.max(fake_B_DIV)
        visuals.append(('output_divergence', fake_B_DIV))
        
        if self.isTrain:
            weight_mask = util.tensor2im(self.weight_mask.data)
            if not self.opt.local_loss:
                weight_mask[mask_edge_coords] = np.max(weight_mask)
            visuals.append(('L2 weight mask', weight_mask))
            

        if self.opt.continent_data:
            continents = util.tensor2im(self.continents.data)
            continents[mask_edge_coords] = np.max(continents)
            visuals.append(('continents', continents))

        return OrderedDict(visuals)


    def get_current_metrics(self):
        # import skimage.io as io
        # import matplotlib.pyplot as plt

        metrics = []

        return OrderedDict(metrics)


    def accumulate_metrics(self, metrics):
        return OrderedDict([])


    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)


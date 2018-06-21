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
        return 'DivInlineModel'

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

        self.sobel_layer_y = torch.nn.Conv2d(1, 1, 3, padding=1)
        self.sobel_layer_y.weight.data = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
        self.sobel_layer_y.weight.requires_grad = False


        self.sobel_layer_x = torch.nn.Conv2d(1, 1, 3, padding=1)
        self.sobel_layer_x.weight.data = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
        self.sobel_layer_x.weight.requires_grad = False


        if len(self.gpu_ids) > 0:
            self.sobel_layer_y.cuda(self.gpu_ids[0])
            self.sobel_layer_x.cuda(self.gpu_ids[0])


        if self.isTrain and opt.num_folders > 1 and self.opt.folder_pred:
            self.netG.inner_layer = get_innermost(self.netG, 'UnetSkipConnectionBlock')
            self.netG.inner_layer.register_forward_hook(save_output_hook)

            # Image size downsampled, times number of filters
            self.folder_fc = torch.nn.Linear(2*opt.fineSize**2 / get_downsample(self.netG)**2 * opt.ngf*8, opt.num_folders)

            if len(self.gpu_ids) > 0:
                self.folder_fc.cuda(self.gpu_ids[0])

        if self.isTrain:
            # Inputs: 3 channels of one-hot input (with chunk missing) + divergence output data
            discrim_input_channels = opt.input_nc + opt.output_nc

            # Add extra channel for mask if we need it
            if not self.opt.no_mask_to_critic:
                discrim_input_channels += 1

            if self.opt.continent_data:
                discrim_input_channels += 1

            self.netDs = [DiscriminatorWGANGP(discrim_input_channels, (256, 512), opt.ndf) for _ in range(self.opt.num_discrims)]

            # Apply is in-place, we don't need to return into anything
            [netD.apply(weights_init) for netD in self.netDs]

            if len(self.gpu_ids) > 0:
                [netD.cuda() for netD in self.netDs]

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

            if self.isTrain:
                [self.load_network(self.netDs[i], 'D_%d' % i, opt.which_epoch) for i in range(len(self.netDs))]


        if self.isTrain:
            # define loss functions

            self.criterionL2 = torch.nn.MSELoss(reduce=False)
            self.criterionCE = torch.nn.NLLLoss2d

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)

            self.optimizer_Ds = [torch.optim.Adam(netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999)) for netD in self.netDs]
            self.optimizers += self.optimizer_Ds
           
            
            # Just a linear decay over the last 100 iterations, by default
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            if self.opt.num_discrims > 0:
                networks.print_network(self.netDs[0])
            print("#discriminators", len(self.netDs))
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

        self.batch_size = input_A.shape[0]

        self.mask_size = input['mask_size'].numpy()[0]
        print('mask size in set_input', self.mask_size)
        self.div_thresh = input['DIV_thresh']
        self.div_min = input['DIV_min']
        self.div_max = input['DIV_max']

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

        if opt.grad_loss:
            self.real_B_DIV_grad_x = self.sobel_layer_x(self.real_B_DIV)
            self.real_B_DIV_grad_y = self.sobel_layer_y(self.real_B_DIV)

            self.fake_B_DIV_grad_x = self.sobel_layer_x(self.fake_B_DIV)
            self.fake_B_DIV_grad_y = self.sobel_layer_y(self.fake_B_DIV)

        scaled_thresh = self.div_thresh.repeat(1, 3) / torch.cat((self.div_max, torch.ones(self.div_max.shape), -self.div_min), dim=1)
        scaled_thresh = scaled_thresh.view(self.fake_B_DIV.shape[0], 3, 1, 1)
        self.fake_B_discrete = (torch.cat((self.fake_B_DIV, torch.zeros(self.fake_B_DIV.shape), -self.fake_B_DIV), dim=1) > scaled_thresh)
        plate = 1 - torch.max(self.fake_B_discrete, dim=1)[0]

        self.fake_B_discrete[:, 1, :, :].copy_(plate)

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

        scaled_thresh = self.div_thresh.repeat(1, 3) / torch.cat((self.DIV_max, torch.ones(self.div_max.shape), -self.div_min), dim=1)
        scaled_thresh = scaled_thresh.view(self.fake_B_DIV.shape[0], 3, 1, 1)
        self.fake_B_discrete = (torch.cat((self.fake_B_DIV, torch.zeros(self.fake_B_DIV.shape), -self.fake_B_DIV), dim=1) > scaled_thresh)
        plate = 1 - torch.max(self.fake_B_discrete, dim=1)[0]

        self.fake_B_discrete[:, 1, :, :].copy_(plate)

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


    def calc_gradient_penalty(self, netD, real_data, fake_data):
        # print "real_data: ", real_data.size(), fake_data.size()
        alpha = torch.rand(real_data.shape[0], 1)
        alpha = alpha.expand(alpha.shape[0], real_data[0, ...].nelement()).contiguous().view(-1, *real_data.shape[1:])
        alpha = alpha.cuda(self.gpu_ids[0]) if len(self.gpu_ids) > 0 else alpha

        interpolates = alpha * fake_data + ((1 - alpha) * real_data)

        if len(self.gpu_ids) > 0:
            interpolates = interpolates.cuda(self.gpu_ids[0])
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        # We have the [0] at the end because grad() returns a tuple with an empty second element, for some reason
        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(self.gpu_ids[0]) if len(self.gpu_ids) > 0 else torch.ones(
                                      disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        
        gradients = gradients.view(gradients.size(0), -1)
        
        # Flattened, so we take the gradient wrt every x (each pixel in each channel)
        # Take mean across the batch
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean(dim=0, keepdim=True)

        return gradient_penalty


    def backward_single_D(self, net_D, cond_data, real_data, fake_data):
        # Fake
        # In this case real_A, the input, is our conditional vector
        fake_AB = torch.cat((cond_data, fake_data), dim=1)
        # stop backprop to the generator by detaching fake_B
        fake_loss = net_D(fake_AB.detach()).mean(dim=0, keepdim=True)
        # self.loss_D2_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((cond_data, real_data), dim=1)
        # Mean across batch
        real_loss = net_D(real_AB).mean(dim=0, keepdim=True)
        # self.loss_D2_real = self.criterionGAN(pred_real, True)

        grad_pen = torch.zeros((1,))
        grad_pen = grad_pen.cuda() if len(self.gpu_ids) > 0 else grad_pen
        grad_pen = torch.autograd.Variable(grad_pen, requires_grad=False)

        if self.opt.which_model_netD == 'wgan-gp':
            grad_pen = self.calc_gradient_penalty(net_D, real_AB.data, fake_AB.data)

        # Combined loss
        # self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5
        loss = fake_loss - real_loss + grad_pen * self.opt.lambda_C

        loss.backward()

        # We could use view, but it looks like it just causes memory overflow
        # return torch.cat((loss, real_loss, fake_loss), dim=0).view(-1, 3, 1)
        output = torch.cat((loss, real_loss, fake_loss, grad_pen), dim=0)
        output = output.unsqueeze(0)
        output = output.unsqueeze(-1)

        return output


    def backward_D(self, net_Ds, optimisers, cond_data, real_data, fake_data):

        for optimiser in optimisers:
            optimiser.zero_grad()

        # We get back full loss, real loss and fake loss, along axis 1
        # Concatenate the results from each discriminator along axis 2
        loss = torch.cat([self.backward_single_D(net_D, cond_data, real_data, fake_data) for net_D in net_Ds], dim=2)

        for optimiser in optimisers:
            optimiser.step()

        # loss[:, 0, :].backward()        

        # We take the different loss tyes (along axis 1) and take their average across all discriminators (axis 2 before selecting index on axis 1)
        output = (torch.mean(loss[:, 0, :], dim=1, keepdim=True),
            torch.mean(loss[:, 1, :], dim=1, keepdim=True),
            torch.mean(loss[:, 2, :], dim=1, keepdim=True),
            torch.mean(loss[:, 3, :], dim=1, keepdim=True))

        return output


    def backward_G(self):
        self.loss_G_GAN = 0
        self.loss_G_L2 = 0

        if self.opt.num_discrims > 0:
            # Conditional data (input with chunk missing + mask) + fake data
            # Remember self.fake_B_discrete is the generator output
            fake_AB = self.real_A_discrete
            
            if not self.opt.no_mask_to_critic:
                fake_AB = torch.cat((fake_AB, self.mask.float()), dim=1)
            
            if self.opt.continent_data:
                fake_AB = torch.cat((fake_AB, self.continents.float()), dim=1)
            
            # Append fake data
            fake_AB = torch.cat((fake_AB, self.fake_B_DIV), dim=1)
        
            # Mean across batch, then across discriminators
            # We only optimise with respect to the fake prediction because
            # the first term (i.e. the real one) is independent of the generator i.e. it is just a constant term
            pred_fake1 = torch.cat([netD(fake_AB).mean(dim=0, keepdim=True) for netD in self.netDs]).mean(dim=0, keepdim=True)
            
            self.loss_G_GAN1 = -pred_fake1
 
            # Trying to incentivise making this big, so it's mistaken for real
            self.loss_G_GAN = self.loss_G_GAN1


        # if we aren't taking local loss, use entire image
        loss_mask = torch.ones(self.mask.shape).byte()
        loss_mask = loss_mask.cuda() if len(self.gpu_ids) > 0 else loss_mask
        loss_mask = torch.autograd.Variable(loss_mask)

        im_dims = self.mask.shape[2:]

        if self.opt.local_loss:
            loss_mask = self.mask.byte()

            # We could maybe sum across channels 2 and 3 to get these dims, once masks are different sizes
            im_dims = self.mask_size, self.mask_size
            # im_dims = (100, 100)
        

        total_pixels = torch.sum(torch.sum(loss_mask > 0, dim=2, keepdim=True), dim=3, keepdim=True).float()

        self.fake_B_DIV_ROI = self.fake_B_DIV.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)

        self.real_B_DIV_ROI = self.real_B_DIV.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)

        self.real_B_discrete_ROI = self.real_B_discrete.masked_select(loss_mask.repeat(1, 3, 1, 1)).view(self.batch_size, 3, *im_dims)
        self.fake_B_discrete_ROI = self.fake_B_discrete.masked_select(loss_mask.repeat(1, 3, 1, 1)).view(self.batch_size, 3, *im_dims)

        self.weight_mask = util.create_weight_mask(self.real_B_discrete_ROI, self.fake_B_discrete_ROI, self.opt.diff_in_numerator)

        self.loss_G_L2_DIV = (self.weight_mask.detach() * self.criterionL2(self.fake_B_DIV_ROI, self.real_B_DIV_ROI)).sum(dim=2).sum(dim=2).mean(dim=0) * self.opt.lambda_A

        self.loss_G_L2 += self.loss_G_L2_DIV

        # self.fake_B_DIV_ROI = self.fake_B_DIV.masked_select(self.mask.byte()).view(self.batch_size, 1, self.mask_size, self.mask_size)
        # self.real_B_DIV_ROI = self.real_B_DIV.masked_select(self.mask.byte()).view(self.batch_size, 1, self.mask_size, self.mask_size)
        
        if self.opt.grad_loss:
            self.real_B_DIV_grad_x = self.real_B_DIV_grad_x.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)
            self.real_B_DIV_grad_y = self.real_B_DIV_grad_y.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)

            self.fake_B_DIV_grad_x = self.fake_B_DIV_grad_x.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)
            self.fake_B_DIV_grad_y = self.fake_B_DIV_grad_y.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)

            grad_x_L2_img =  self.criterionL2(self.fake_B_DIV_grad_x, self.real_B_DIV_grad_x.detach())
            grad_y_L2_img =  self.criterionL2(self.fake_B_DIV_grad_y, self.real_B_DIV_grad_y.detach())

            if self.opt.weighted_grad:
                grad_x_L2_img = self.weight_mask.detach() * grad_x_L2_img
                grad_y_L2_img = self.weight_mask.detach() * grad_y_L2_img

            self.loss_L2_DIV_grad_x = (grad_x_L2_img).sum(dim=2).sum(dim=2).mean(dim=0)
            self.loss_L2_DIV_grad_y = (grad_y_L2_img).sum(dim=2).sum(dim=2).mean(dim=0)

            print("Adding gradient losses")
            self.loss_G_L2 += self.loss_L2_DIV_grad_x
            self.loss_G_L2 += self.loss_L2_DIV_grad_y


        self.loss_G = self.loss_G_GAN + self.loss_G_L2

        if self.isTrain and self.opt.num_folders > 1 and self.opt.folder_pred:
            ce_fun = self.criterionCE()
            self.folder_pred_CE = ce_fun(self.fake_folder, self.real_folder) * self.opt.lambda_D

            self.loss_G += self.folder_pred_CE

        self.loss_G.backward()


    def optimize_parameters(self, **kwargs):
        # Doesn't do anything with discriminator, just populates input (conditional), 
        # target and generated data in object
        self.forward()

        if self.opt.num_discrims > 0:
            cond_data = self.real_A_discrete

            if not self.opt.no_mask_to_critic:
                cond_data = torch.cat((cond_data, self.mask.float()), dim=1)
 
            if self.opt.continent_data:
                cond_data = torch.cat((cond_data, self.continents.float()), dim=1)



            self.loss_D, self.loss_D_real, self.loss_D_fake, self.loss_D_grad_pen = self.backward_D(self.netDs, self.optimizer_Ds,
                cond_data,
                self.real_B_DIV, self.fake_B_DIV)

        step_no = kwargs['step_no']
        if ((step_no < self.opt.high_iter*25 and step_no % self.opt.high_iter == 0) or (step_no >= self.opt.high_iter*25 and step_no % self.opt.low_iter == 0)) or self.opt.num_discrims == 0:
            if step_no % 10 == 0:
                self.optimizer_G.zero_grad()

            self.backward_G()
        
            if step_no % 10 == 0:
                self.optimizer_G.step()
        

    def get_current_errors(self):
        errors = [
            ('G', self.loss_G.data.item()),
            ('G_L2', self.loss_G_L2.data.item()),
            ('G_L2_DIV', self.loss_G_L2_DIV.data.item()),
            ]

        if self.opt.grad_loss:
            errors += [
                ('G_L2_grad_x', self.loss_L2_DIV_grad_x.data.item()),
                ('G_L2_grad_y', self.loss_L2_DIV_grad_y.data.item())
                ]

        if self.opt.num_discrims > 0:
            errors += [
                ('G_GAN_D', self.loss_G_GAN.data[0]),
                ('D_real', self.loss_D_real.data[0]),
                ('D_fake', self.loss_D_fake.data[0]),
                ('D_grad_pen', self.loss_D_grad_pen.data[0])
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

        fake_B_discrete = util.tensor2im(self.fake_B_discrete.data)
        fake_B_discrete[mask_edge_coords] = np.max(fake_B_discrete)
        visuals.append(('output_discrete', fake_B_discrete))

        
        if self.opt.grad_loss:
            real_B_DIV_grad_x = util.tensor2im(self.real_B_DIV_grad_x.data)
            visuals.append(('ground_truth_x_gradient', real_B_DIV_grad_x))

            real_B_DIV_grad_y = util.tensor2im(self.real_B_DIV_grad_y.data)
            visuals.append(('ground_truth_y_gradient', real_B_DIV_grad_y))

            fake_B_DIV_grad_x = util.tensor2im(self.fake_B_DIV_grad_x.data)
            visuals.append(('output_x_gradient', fake_B_DIV_grad_x))

            fake_B_DIV_grad_y = util.tensor2im(self.fake_B_DIV_grad_y.data)
            visuals.append(('output_y_gradient', fake_B_DIV_grad_y))
            
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
        
        for i in range(len(self.netDs)):
            self.save_network(self.netDs[i], 'D_%d' % i, label, self.gpu_ids)


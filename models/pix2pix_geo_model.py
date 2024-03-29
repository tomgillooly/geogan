import torch
import torchvision.transforms as transforms
import torch.autograd as autograd
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from .second_head import SecondHead

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


class Pix2PixGeoModel(BaseModel):
    def name(self):
        return 'Pix2PixGeoModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

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
            self.folder_fc = torch.nn.Linear(2*opt.fineSize**2 / get_downsample(self.netG)**2 * opt.ngf*8, opt.num_folders)

            if len(self.gpu_ids) > 0:
                self.folder_fc.cuda(self.gpu_ids[0])

        if not self.opt.discrete_only or self.opt.div_only:
            if self.opt.deep_div:
                self.netG_DIV = SecondHead(self.netG, 1, 'ConvTranspose2d', self.opt.div_depth, 'Conv2d')
            else:
                self.netG_DIV = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=opt.output_nc, out_channels=1, kernel_size=1),
                    )
            
            if len(self.gpu_ids) > 0:
                self.netG_DIV.cuda(self.gpu_ids[0])
        
            if not self.opt.div_only:
                self.netG_Vx = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=opt.output_nc, out_channels=1, kernel_size=1),
                    )
                self.netG_Vy = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels=opt.output_nc, out_channels=1, kernel_size=1),
                    )

                if len(self.gpu_ids) > 0:
                    self.netG_Vx.cuda(self.gpu_ids[0])
                    self.netG_Vy.cuda(self.gpu_ids[0])


        def get_discriminator(input_channels):
            # def create_WGAN_GP():

            if self.opt.which_model_netD == 'wgan-gp':
                return DiscriminatorWGANGP(input_channels, (256, 512), opt.ndf)

            else:
            # def create_PatchGAN():
                use_sigmoid = opt.no_lsgan
                return networks.define_D(input_channels, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm,
                                          use_sigmoid, opt.init_type, self.gpu_ids)
            
                # return create_WGAN_GP
                # return create_PatchGAN


        if self.isTrain:
            # Inputs: 3 channels of one-hot input (with chunk missing) + discrete output data
            discrim_input_channels = opt.input_nc + opt.output_nc

            # Add extra channel for mask if we need it
            if not self.opt.no_mask_to_critic:
                discrim_input_channels += 1

            if self.opt.continent_data:
                discrim_input_channels += 1

            self.netD1s = [get_discriminator(discrim_input_channels) for _ in range(self.opt.num_discrims)]

            # Apply is in-place, we don't need to return into anything
            [netD1.apply(weights_init) for netD1 in self.netD1s]

            if len(self.gpu_ids) > 0:
                [netD.cuda() for netD in self.netD1s]

            if not self.opt.discrete_only or self.opt.div_only:
                # Inputs: 3 channels of one-hot input (with chunk missing)
                discrim_input_channels = opt.input_nc

                # Add extra channel for mask if we need it
                if not self.opt.no_mask_to_critic:
                    discrim_input_channels += 1

                if self.opt.continent_data:
                    discrim_input_channels += 1

                # Just divergence
                if self.opt.div_only:
                    discrim_input_channels += 1
                # Divergence and two velocity images
                else:
                    discrim_input_channels += 3


                # 3 channels of one-hot input (with chunk missing) + mask + DIV, Vx, Vy
                self.netD2s = [get_discriminator(discrim_input_channels) for _ in range(self.opt.num_discrims)]

                [netD2.apply(weights_init) for netD2 in self.netD2s]

                if len(self.gpu_ids) > 0:
                    [netD.cuda() for netD in self.netD2s]


        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)


            if not self.opt.discrete_only or self.opt.div_only:
                try:
                    self.load_network(self.netG_DIV, 'G_DIV', opt.which_epoch)

                    if not self.opt.div_only:
                        self.load_network(self.netG_DIV, 'G_Vx', opt.which_epoch)
                        self.load_network(self.netG_DIV, 'G_Vy', opt.which_epoch)
                except FileNotFoundError:
                    self.opt.discrete_only = True
            
            if self.isTrain:
                [self.load_network(self.netD1s[i], 'D1_%d' % i, opt.which_epoch) for i in range(len(self.netD1s))]

                if not self.opt.discrete_only or self.opt.div_only:
                    [self.load_network(self.netD2s[i], 'D2_%d' % i, opt.which_epoch) for i in range(len(self.netD2s))]

        if self.isTrain:
            # define loss functions

            if opt.which_model_netD == 'wgan-gp':
                def batch_mean(data, data_is_real):
                    return data.mean(dim=0, keepdim=True) * (-1 if data_is_real else 1)
                self.criterionGAN = batch_mean
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan,
                    tensor=self.Tensor)

            self.criterionL2 = torch.nn.MSELoss(size_average=False)
            self.criterionCE = torch.nn.NLLLoss2d

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizer_D1s = [torch.optim.Adam(netD1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999)) for netD1 in self.netD1s]
            self.optimizers += self.optimizer_D1s
            
            if not self.opt.discrete_only or self.opt.div_only:
                self.optimizer_G_DIV = torch.optim.Adam(self.netG_DIV.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizers.append(self.optimizer_G_DIV)
                self.optimizer_D2s = [torch.optim.Adam(netD2.parameters(),
                                                    lr=opt.lr, betas=(opt.beta1, 0.999)) for netD2 in self.netD2s]
                self.optimizers += self.optimizer_D2s
                
                if not self.opt.div_only:
                    self.optimizer_G_Vx = torch.optim.Adam(self.netG_Vx.parameters(),
                                                        lr=opt.lr, betas=(opt.beta1, 0.999))
                    self.optimizer_G_Vy = torch.optim.Adam(self.netG_Vy.parameters(),
                                                        lr=opt.lr, betas=(opt.beta1, 0.999))
                    self.optimizers.append(self.optimizer_G_Vx)
                    self.optimizers.append(self.optimizer_G_Vy)

            # Just a linear decay over the last 100 iterations, by default
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            if self.opt.num_discrims > 0:
                networks.print_network(self.netD1s[0])
            print("#discriminators", len(self.netD1s))
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
        input_A_Vx  = input['A_Vx' if AtoB else 'B_Vx']
        input_B_Vx  = input['B_Vx' if AtoB else 'A_Vx']
        input_A_Vy  = input['A_Vy' if AtoB else 'B_Vy']
        input_B_Vy  = input['B_Vy' if AtoB else 'A_Vy']
        mask        = input['mask']

        if self.opt.continent_data:
            continents = input['cont']
        
        if len(self.gpu_ids) > 0:
            input_A     = input_A.cuda(self.gpu_ids[0], async=True)
            input_B     = input_B.cuda(self.gpu_ids[0], async=True)
            input_A_DIV = input_A_DIV.cuda(self.gpu_ids[0], async=True)
            input_B_DIV = input_B_DIV.cuda(self.gpu_ids[0], async=True)
            input_A_Vx  = input_A_Vx.cuda(self.gpu_ids[0], async=True)
            input_B_Vx  = input_B_Vx.cuda(self.gpu_ids[0], async=True)
            input_A_Vy  = input_A_Vy.cuda(self.gpu_ids[0], async=True)
            input_B_Vy  = input_B_Vy.cuda(self.gpu_ids[0], async=True)
            mask        = mask.cuda(self.gpu_ids[0], async=True)
            
            if self.opt.continent_data:
                continents = continents.cuda(self.gpu_ids[0], async=True)
        
        self.input_A        = input_A
        self.input_B        = input_B
        self.input_A_DIV    = input_A_DIV
        self.input_B_DIV    = input_B_DIV
        self.input_A_Vx     = input_A_Vx
        self.input_B_Vx     = input_B_Vx
        self.input_A_Vy     = input_A_Vy
        self.input_B_Vy     = input_B_Vy
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


        if not self.opt.discrete_only or self.opt.div_only:
            # Continuous divergence map with chunk missing
            self.real_A_DIV = torch.autograd.Variable(self.input_A_DIV)
            
            # Complete continuous divergence map
            self.real_B_DIV = torch.autograd.Variable(self.input_B_DIV)

            if not self.opt.div_only:
                # Vector fields with chunk missing
                self.real_A_Vx = torch.autograd.Variable(self.input_A_Vx)
                self.real_A_Vy = torch.autograd.Variable(self.input_A_Vy)
                # Complete vector field
                self.real_B_Vx = torch.autograd.Variable(self.input_B_Vx)
                self.real_B_Vy = torch.autograd.Variable(self.input_B_Vy)

        # Mask of inpainted region
        self.mask = torch.autograd.Variable(self.mask, requires_grad=False)

        if self.opt.continent_data:
            self.continents = torch.autograd.Variable(self.continent_img)
        
        # Produces three channel output with class probability assignments
        # Input is one-hot image with chunk missing, conditional data is mask
        self.G_input = torch.cat((self.real_A_discrete, self.mask.float()), dim=1)

        if self.opt.continent_data:
            self.G_input = torch.cat((self.G_input, self.continents.float()), dim=1)


        self.fake_B_discrete = self.netG(self.G_input)

        if self.opt.isTrain and self.opt.num_folders > 1 and self.opt.folder_pred:
            self.fake_folder = self.folder_fc(self.netG.inner_layer.output.view(self.batch_size, -1))
            self.fake_folder = torch.nn.functional.log_softmax(self.fake_folder, dim=1)
        

        if not self.opt.discrete_only or self.opt.div_only:
            # Create continuous divergence field from class probabilities
            self.fake_B_DIV = self.netG_DIV(self.fake_B_discrete)

            if not self.opt.div_only:
                # Create vector field from class probabilities
                self.fake_B_Vx = self.netG_Vx(self.fake_B_discrete)
                self.fake_B_Vy = self.netG_Vy(self.fake_B_discrete)
        
        # Find log probabilities for NLL step in backprop
        self.fake_B_softmax = F.log_softmax(self.fake_B_discrete, dim=1)
        # Apply max and normalise to get -1 to 1 range
        self.fake_B_classes = torch.max(self.fake_B_softmax, dim=1, keepdim=True)[1]

        self.real_B_softmax = F.log_softmax(self.real_B_discrete, dim=1)
        self.real_B_classes = torch.max(self.real_B_softmax, dim=1, keepdim=False)[1]

        self.fake_B_one_hot = torch.zeros(self.fake_B_discrete.shape)
        
        # scatter_ is a function that fills based on the indices in fake_B_classes, along the specified axis
        # Remember axis 1 is the channel axis
        self.fake_B_one_hot.scatter_(1, self.fake_B_classes.data.cpu(), 1.0)

    # no backprop gradients
    def test(self):
        self.real_A_discrete = torch.autograd.Variable(self.input_A, volatile=True)
        self.real_B_discrete = torch.autograd.Variable(self.input_B, requires_grad=False)

        if not self.opt.discrete_only or self.opt.div_only:
            self.real_A_DIV = torch.autograd.Variable(self.input_A_DIV)
            self.real_B_DIV = torch.autograd.Variable(self.input_B_DIV)

            if not self.opt.div_only:
                self.real_A_Vx = torch.autograd.Variable(self.input_A_Vx)
                self.real_A_Vy = torch.autograd.Variable(self.input_A_Vy)
                self.real_B_Vx = torch.autograd.Variable(self.input_B_Vx)
                self.real_B_Vy = torch.autograd.Variable(self.input_B_Vy)
        
        self.mask = torch.autograd.Variable(self.mask)
        
        if self.opt.continent_data:
            self.continents = torch.autograd.Variable(self.continent_img)

        # mask_var = Variable(self.mask.float(), volatile=True)
        self.G_input = torch.cat((self.real_A_discrete, self.mask.float()), dim=1)

        if self.opt.continent_data:
            self.G_input = torch.cat((self.G_input, self.continents.float()), dim=1)
        
        self.fake_B_discrete = self.netG(self.G_input)

        if not self.opt.discrete_only or self.opt.div_only:
            self.fake_B_DIV = self.netG_DIV(self.fake_B_discrete)

            if not self.opt.div_only:
                self.fake_B_Vx = self.netG_Vx(self.fake_B_discrete)
                self.fake_B_Vy = self.netG_Vy(self.fake_B_discrete)
        
        self.fake_B_softmax = F.log_softmax(self.fake_B_discrete, dim=1)
        self.fake_B_classes = torch.max(self.fake_B_softmax, dim=1, keepdim=True)[1]

        self.real_B_softmax = F.log_softmax(self.real_B_discrete, dim=1)
        self.real_B_classes = torch.max(self.real_B_softmax, dim=1, keepdim=True)[1]

        self.fake_B_one_hot = torch.zeros(self.fake_B_discrete.shape)

        self.fake_B_one_hot.scatter_(1, self.fake_B_classes.data.cpu(), 1.0)

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
        fake_loss = self.criterionGAN(net_D(fake_AB.detach()), False)
        # self.loss_D2_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((cond_data, real_data), dim=1)
        # Mean across batch
        real_loss = self.criterionGAN(net_D(real_AB), True)
        # self.loss_D2_real = self.criterionGAN(pred_real, True)

        grad_pen = torch.zeros((1,))
        grad_pen = grad_pen.cuda() if len(self.gpu_ids) > 0 else grad_pen
        grad_pen = torch.autograd.Variable(grad_pen, requires_grad=False)

        if self.opt.which_model_netD == 'wgan-gp':
            grad_pen = self.calc_gradient_penalty(net_D, real_AB.data, fake_AB.data)

        # Combined loss
        # self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5
        loss = fake_loss + real_loss + grad_pen * self.opt.lambda_C

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
        # First, G(A) should fake the discriminator
        # Note that we don't detach here because we DO want to backpropagate
        # to the generator this time

        self.loss_G_GAN = 0
            
        if self.opt.num_discrims > 0:
            # Conditional data (input with chunk missing + mask) + fake data
            # Remember self.fake_B_discrete is the generator output
            fake_AB = self.real_A_discrete
            
            if not self.opt.no_mask_to_critic:
                fake_AB = torch.cat((fake_AB, self.mask.float()), dim=1)
            
            if self.opt.continent_data:
                fake_AB = torch.cat((fake_AB, self.continents.float()), dim=1)
            
            # Append fake data
            fake_AB = torch.cat((fake_AB, self.fake_B_discrete), dim=1)
        
            # Mean across batch, then across discriminators
            # We only optimise with respect to the fake prediction because
            # the first term (i.e. the real one) is independent of the generator i.e. it is just a constant term
            pred_fake1 = torch.cat([self.criterionGAN(netD1(fake_AB), True) for netD1 in self.netD1s]).mean(dim=0, keepdim=True)
            
            self.loss_G_GAN1 = pred_fake1
 
            # Trying to incentivise making this big, so it's mistaken for real
            self.loss_G_GAN = self.loss_G_GAN1
        
            if not self.opt.discrete_only or self.opt.div_only:
                # Conditional data (input with chunk missing + mask) + fake DIV, Vx and Vy data
                fake_AB = torch.cat((self.real_A_discrete,), dim=1)
 
                if not self.opt.no_mask_to_critic:
                    fake_AB = torch.cat((fake_AB, self.mask.float()), dim=1)
 
                if self.opt.continent_data:
                    fake_AB = torch.cat((fake_AB, self.continents.float()), dim=1)
 
                fake_AB = torch.cat((fake_AB, self.fake_B_DIV), dim=1)

                if not self.opt.div_only:
                    fake_AB = torch.cat((fake_AB, self.fake_B_Vx, self.fake_B_Vy), dim=1)


                # Mean across batch, then across discriminators
                pred_fake2 = torch.cat([self.criterionGAN(netD2(fake_AB), True) for netD2 in self.netD2s]).mean(dim=0, keepdim=True)

                self.loss_G_GAN2 = pred_fake2

                self.loss_G_GAN += self.loss_G_GAN2

        self.loss_G = self.loss_G_GAN 
        
        # if we aren't taking local loss, use entire image
        loss_mask = torch.ones(self.mask.shape).byte()

        im_dims = self.mask.shape[2:]

        if self.opt.local_loss:
            loss_mask = self.mask.byte()

            # We could maybe sum across channels 2 and 3 to get these dims, once masks are different sizes
            # im_dims = self.mask_size_y[0], self.mask_size_x[0]
            im_dims = (100, 100)
        
        loss_mask = loss_mask.cuda() if len(self.gpu_ids) > 0 else loss_mask
        loss_mask = torch.autograd.Variable(loss_mask)


        if not self.opt.discrete_only or self.opt.div_only:
            self.fake_B_DIV_ROI = self.fake_B_DIV.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)
            self.real_B_DIV_ROI = self.real_B_DIV.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)

            self.loss_G_L2_DIV = self.criterionL2(
                self.fake_B_DIV_ROI,
                self.real_B_DIV_ROI) * self.opt.lambda_A

            self.loss_G_L2 = self.loss_G_L2_DIV

            if not self.opt.div_only:
                self.fake_B_Vx_ROI = self.fake_B_Vx.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)
                self.real_B_Vx_ROI = self.real_B_Vx.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)
                
                self.fake_B_Vy_ROI = self.fake_B_Vy.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)
                self.real_B_Vy_ROI = self.real_B_Vy.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)
                self.loss_G_L2_Vx = self.criterionL2(
                    self.fake_B_Vx_ROI,
                    self.real_B_Vx_ROI) * self.opt.lambda_A
                self.loss_G_L2_Vy = self.criterionL2(
                    self.fake_B_Vy_ROI,
                    self.real_B_Vy_ROI) * self.opt.lambda_A

                self.loss_G_L2 += self.loss_G_L2_Vx + self.loss_G_L2_Vy


            self.loss_G += self.loss_G_L2


        self.fake_B_discrete_ROI = self.fake_B_discrete.masked_select(loss_mask.repeat(1, 3, 1, 1)).view(
                self.batch_size, 3, *im_dims)
        self.real_B_discrete_ROI = self.real_B_discrete.masked_select(loss_mask.repeat(1, 3, 1, 1)).view(
                self.batch_size, 3, *im_dims)
        self.real_B_classes_ROI = self.real_B_classes.masked_select(loss_mask.squeeze()).view(
                self.batch_size, *im_dims)

        total_pixels = 1.0 * im_dims[0] * im_dims[1]
        
        num_ridge_pixels = torch.sum(torch.sum(self.real_B_discrete_ROI[:, 0, :, :], dim=1), dim=1)
        num_plate_pixels = torch.sum(torch.sum(self.real_B_discrete_ROI[:, 1, :, :], dim=1), dim=1)
        num_subduction_pixels = torch.sum(torch.sum(self.real_B_discrete_ROI[:, 2, :, :], dim=1), dim=1)

        ridge_weight = 1.0 - num_ridge_pixels / total_pixels
        plate_weight = 1.0 - num_plate_pixels / total_pixels
        subduction_weight = 1.0 - num_subduction_pixels / total_pixels

        ridge_weight = ridge_weight.mean(dim=0, keepdim=True)
        plate_weight = plate_weight.mean(dim=0, keepdim=True)
        subduction_weight = subduction_weight.mean(dim=0, keepdim=True)
        
        weights = torch.cat((ridge_weight, plate_weight, subduction_weight))

        ce_fun = self.criterionCE(weight=weights)

        # print(fake_B_discrete_masked)
        # print(real_B_classes_masked)

        self.loss_G_CE = ce_fun(F.log_softmax(self.fake_B_discrete_ROI, dim=1),
            self.real_B_classes_ROI) * self.opt.lambda_B

        self.loss_G += self.loss_G_CE

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



            self.loss_D1, self.loss_D1_real, self.loss_D1_fake, self.loss_D1_grad_pen = self.backward_D(self.netD1s, self.optimizer_D1s,
                cond_data,
                self.real_B_discrete, self.fake_B_discrete)

            if not self.opt.discrete_only or self.opt.div_only:
                # Conditional data (input with chunk missing + mask) + fake DIV, Vx and Vy data
                real_data = torch.cat((self.real_B_DIV,), dim=1)
                fake_data = torch.cat((self.fake_B_DIV,), dim=1)

                if not self.opt.div_only:
                    real_data = torch.cat((real_data, self.real_B_Vx, self.real_B_Vy), dim=1)
                    fake_data = torch.cat((fake_data, self.fake_B_Vx, self.fake_B_Vy), dim=1)

                self.loss_D2, self.loss_D2_real, self.loss_D2_fake, self.loss_D2_grad_pen = self.backward_D(self.netD2s, self.optimizer_D2s,
                    cond_data,
                    real_data, 
                    fake_data)


        step_no = kwargs['step_no']
        if (step_no < self.opt.high_iter*25 and step_no % self.opt.high_iter == 0) or (step_no >= self.opt.high_iter*25 and step_no % self.opt.low_iter == 0):
            self.optimizer_G.zero_grad()

            if not self.opt.discrete_only or self.opt.div_only:
                self.optimizer_G_DIV.zero_grad()

                if not self.opt.div_only:
                    self.optimizer_G_Vx.zero_grad()
                    self.optimizer_G_Vy.zero_grad()
            
            self.backward_G()
            
            self.optimizer_G.step()
            
            if not self.opt.discrete_only or self.opt.div_only:
                self.optimizer_G_DIV.step()

                if not self.opt.div_only:
                    self.optimizer_G_Vx.step()
                    self.optimizer_G_Vy.step()


    def get_current_errors(self):
        errors = [
            ('G', self.loss_G.data[0]),
            ('G_CE', self.loss_G_CE.data.item()),
        ]

        if self.opt.num_discrims > 0:
            errors += [
                ('G_GAN_D1', -self.loss_G_GAN1.data[0]),
                ('D1_real', -self.loss_D1_real.data[0]),
                ('D1_fake', self.loss_D1_fake.data[0]),
                ('D1_grad_pen', self.loss_D1_grad_pen.data[0])
            ]

        if not self.opt.discrete_only or self.opt.div_only:
            errors += [
                ('G_L2', self.loss_G_L2.data.item()),
            ]

            if self.opt.num_discrims > 0:
                errors += [
                    ('G_GAN_D2', -self.loss_G_GAN2.data[0]),
                    ('D2_real', -self.loss_D2_real.data[0]),
                    ('D2_fake', self.loss_D2_fake.data[0]),
                    ('D2_grad_pen', self.loss_D2_grad_pen.data[0])
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

        fake_B_discrete = util.tensor2im(F.log_softmax(self.fake_B_discrete, dim=1).data)
        fake_B_discrete[mask_edge_coords] = np.max(fake_B_discrete)
        visuals.append(('output_softmax', fake_B_discrete))

        fake_B_one_hot = util.tensor2im(self.fake_B_one_hot)
        fake_B_one_hot[mask_edge_coords] = np.max(fake_B_one_hot)
        visuals.append(('output_one_hot', fake_B_one_hot))


        if not self.opt.discrete_only or self.opt.div_only:
            real_A_DIV = util.tensor2im(self.real_A_DIV.data)
            real_A_DIV[mask_edge_coords] = np.max(real_A_DIV)
            visuals.append(('input_divergence', real_A_DIV))

            real_B_DIV = util.tensor2im(self.real_B_DIV.data)
            real_B_DIV[mask_edge_coords] = np.max(real_B_DIV)
            visuals.append(('ground_truth_divergence', real_B_DIV))

            fake_B_DIV = util.tensor2im(self.fake_B_DIV.data)
            fake_B_DIV[mask_edge_coords] = np.max(fake_B_DIV)
            visuals.append(('output_divergence', fake_B_DIV))
            
            if not self.opt.div_only:
                real_A_Vx = util.tensor2im(self.real_A_Vx.data)
                real_A_Vx[mask_edge_coords] = np.max(real_A_Vx)
                visuals.append(('input_Vx', real_A_Vx))

                real_A_Vy = util.tensor2im(self.real_A_Vy.data)
                real_A_Vy[mask_edge_coords] = np.max(real_A_Vy)
                visuals.append(('input_Vy', real_A_Vy))

                real_B_Vx = util.tensor2im(self.real_B_Vx.data)
                real_B_Vx[mask_edge_coords] = np.max(real_B_Vx)
                visuals.append(('ground_truth_Vx', real_B_Vx))

                real_B_Vy = util.tensor2im(self.real_B_Vy.data)
                real_B_Vy[mask_edge_coords] = np.max(real_B_Vy)
                visuals.append(('ground_truth_Vy', real_B_Vy))

                fake_B_Vx = util.tensor2im(self.fake_B_Vx.data)
                fake_B_Vx[mask_edge_coords] = np.max(fake_B_Vx)
                visuals.append(('output_Vx', fake_B_Vx))

                fake_B_Vy = util.tensor2im(self.fake_B_Vy.data)
                fake_B_Vy[mask_edge_coords] = np.max(fake_B_Vy)
                visuals.append(('output_Vy', fake_B_Vy))

        if self.opt.continent_data:
            continents = util.tensor2im(self.continents.data)
            continents[mask_edge_coords] = np.max(continents)
            visuals.append(('continents', continents))

        return OrderedDict(visuals)


    def get_current_metrics(self):
        # import skimage.io as io
        # import matplotlib.pyplot as plt

        metrics = []

        precisions = []
        recalls = []

        inpaint_precisions = []
        inpaint_recalls = []

        eps = np.finfo(float).eps

        def get_stats(real, fake):
            num_true_pos = np.sum(np.logical_and(fake_channel, fake_channel == real_channel).ravel())
            num_true_neg = np.sum(np.logical_and(1-fake_channel, fake_channel == real_channel).ravel())
            num_false_pos = np.sum((fake_channel > real_channel).ravel())
            num_false_neg = np.sum((fake_channel < real_channel).ravel())

            return (num_true_pos, num_true_neg, num_false_pos, num_false_neg)

        mask_coords = np.where(self.mask.data.numpy().squeeze())
        mask_tl =(mask_coords[0][0], mask_coords[1][0])
        mask_br =(mask_coords[0][-1], mask_coords[1][-1])

        # d_h_recall = 0.0
        # d_h_precision = 0.0
        # d_h_symmetric = 0.0

        d_h_recall_exc = 0.0
        d_h_precision_exc = 0.0
        d_h_symmetric_exc = 0.0


        # for c in np.unique(self.real_B_classes.data.numpy()):
        for c in [0, 2]:
            fake_channel = self.fake_B_one_hot.numpy().squeeze()[c]
            real_channel = self.real_B_discrete.data.numpy().squeeze()[c]
           
            fake_channel = fake_channel[mask_tl[0]:mask_br[0], mask_tl[1]:mask_br[1]]
            real_channel = real_channel[mask_tl[0]:mask_br[0], mask_tl[1]:mask_br[1]]

            # fake_coords = np.array(np.where(fake_channel)).T
            # real_coords = np.array(np.where(real_channel)).T

            # if not fake_coords.any() or not real_coords.any():
            #     mask_diagonal = euclidean(mask_br, mask_tl)
            #     d_h_fr = mask_diagonal
            #     d_h_rf = mask_diagonal
            #     d_h_s = mask_diagonal
            # elif fake_coords.any() and real_coords.any():
            #     d_h_fr, i1_fr, i2_fr = directed_hausdorff(fake_coords, real_coords)
            #     d_h_rf, i1_rf, i2_rf = directed_hausdorff(real_coords, fake_coords)

                # d_h_s = max(d_h_s, max(d_h_fr, d_h_rf))

            # d_h_p, d_h_r, d_h_s = get_hausdorff(fake_channel, real_channel)
            
            d_h_p_exc, d_h_r_exc, d_h_s_exc = get_hausdorff_exc(fake_channel, real_channel)

            # d_h_recall = max(d_h_recall, d_h_r)
            # d_h_precision = max(d_h_precision, d_h_p)
            # d_h_symmetric = max(d_h_symmetric, d_h_s)

            d_h_recall_exc = max(d_h_recall_exc, d_h_r_exc)
            d_h_precision_exc = max(d_h_precision_exc, d_h_p_exc)
            d_h_symmetric_exc = max(d_h_symmetric_exc, d_h_s_exc)

        # metrics.append(('Hausdorff distance (R)', d_h_recall))
        # metrics.append(('Hausdorff distance (P)', d_h_precision))
        # metrics.append(('Hausdorff distance (S)', d_h_symmetric))

        metrics.append(('Hausdorff distance (R - exc)', d_h_recall_exc))
        metrics.append(('Hausdorff distance (P - exc)', d_h_precision_exc))
        metrics.append(('Hausdorff distance (S - exc)', d_h_symmetric_exc))

        # metrics.append(('Average precision', np.mean(precisions)))
        # metrics.append(('Average recall', np.mean(recalls)))

        # metrics.append(('Average precision (inpainted region)', np.mean(inpaint_precisions)))
        # metrics.append(('Average recall (inpainted region)', np.mean(inpaint_recalls)))

        return OrderedDict(metrics)


    def accumulate_metrics(self, metrics):
        # d_h_recall = []
        # d_h_precision = []
        # d_h_s = []
        
        d_h_recall_exc = []
        d_h_precision_exc = []
        d_h_s_exc = []

        ot_r = []
        ot_s = []

        for metric in metrics:
            # d_h_recall.append(metric['Hausdorff distance (R)'])
            # d_h_precision.append(metric['Hausdorff distance (P)'])
            # d_h_s.append(metric['Hausdorff distance (S)'])
            
            d_h_recall_exc.append(metric['Hausdorff distance (R - exc)'])
            d_h_precision_exc.append(metric['Hausdorff distance (P - exc)'])
            d_h_s_exc.append(metric['Hausdorff distance (S - exc)'])
            
            # ot_r.append(metric['EMD class 0'])
            # ot_s.append(metric['EMD class 2'])


        return OrderedDict([
            # ('Hausdorff distance (R)', np.mean(d_h_recall)),
            # ('Hausdorff distance (P)', np.mean(d_h_precision)),
            # ('Hausdorff distance (S)', np.mean(d_h_s)),
            
            ('Hausdorff distance (R - exc)', np.mean(d_h_recall_exc)),
            ('Hausdorff distance (P - exc)', np.mean(d_h_precision_exc)),
            ('Hausdorff distance (S - exc)', np.mean(d_h_s_exc)),
            # ('EMD class 0', np.mean(ot_r)),
            # ('EMD class 2', np.mean(ot_s)),
            ])


    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)

        if not self.opt.discrete_only or self.opt.div_only:
            self.save_network(self.netG_DIV, 'G_DIV', label, self.gpu_ids)

            if not self.opt.div_only:
                self.save_network(self.netG_Vx, 'G_Vx', label, self.gpu_ids)
                self.save_network(self.netG_Vy, 'G_Vy', label, self.gpu_ids)

        for i in range(len(self.netD1s)):
            self.save_network(self.netD1s[i], 'D1_%d' % i, label, self.gpu_ids)
            if not self.opt.discrete_only or self.opt.div_only:
                self.save_network(self.netD2s[i], 'D2_%d' % i, label, self.gpu_ids)

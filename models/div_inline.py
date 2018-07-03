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


def wgan_criterionGAN(loss, real_label):
    return loss.mean(dim=0, keepdim=True) * (-1 if real_label else 1)


def hinge_criterionGAN(loss, real_label):
    if real_label:
        return torch.nn.ReLU()(1.0 - loss).mean()
    else:
        return torch.nn.ReLU()(1.0 + loss).mean()


class DivInlineModel(BaseModel):
    def name(self):
        return 'DivInlineModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        self.D_has_run = False

        self.p = GeoPickler('')

        # load/define networks
        # Input channels = 3 channels for input one-hot map + mask
        input_channels = opt.input_nc

        if self.opt.continent_data:
            input_channels += 1

        self.netG = networks.define_G(input_channels, opt.output_nc+2, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)


        if self.opt.grad_loss:
            self.sobel_layer_y = torch.nn.Conv2d(1, 1, 3, padding=1)
            self.sobel_layer_y.weight.data = torch.FloatTensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).unsqueeze(0).unsqueeze(0)
            self.sobel_layer_y.weight.requires_grad = False


            self.sobel_layer_x = torch.nn.Conv2d(1, 1, 3, padding=1)
            self.sobel_layer_x.weight.data = torch.FloatTensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).unsqueeze(0).unsqueeze(0)
            self.sobel_layer_x.weight.requires_grad = False


            if len(self.gpu_ids) > 0:
                self.sobel_layer_y.cuda(self.gpu_ids[0])
                self.sobel_layer_x.cuda(self.gpu_ids[0])


        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            # Inputs: 3 channels of one-hot input (with chunk missing) + divergence output data + mask
            discrim_input_channels = opt.output_nc

            if not opt.no_mask_to_critic:
                discrim_input_channels += 1


            if opt.local_critic:
                self.critic_im_size = (64, 64)
            else:
                self.critic_im_size = (256, 256)

            self.netD = networks.define_D(discrim_input_channels, opt.ndf, opt.which_model_netD, opt.n_layers_D, 
                opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, critic_im_size=self.critic_im_size)
            

            if len(self.gpu_ids) > 0:
                self.netD.cuda()

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)

            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)


        if self.isTrain:
            # define loss functions

            self.criterionL2 = torch.nn.MSELoss(reduce=False)
            # self.criterionCE = torch.nn.NLLLoss2d()
            self.criterionCE = torch.nn.CrossEntropyLoss(reduce=False)

            if self.opt.use_hinge:
                self.criterionGAN = hinge_criterionGAN
            elif self.opt.which_model_netD == 'wgan-gp' or self.opt.which_model_netD == 'self-attn':
                self.criterionGAN = wgan_criterionGAN
            else:
                self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []

            if opt.optim_type == 'adam':
                optim = torch.optim.Adam
                G_optim_kwargs = {'lr': opt.g_lr, 'betas': (opt.g_beta1, 0.999)}
                D_optim_kwargs = {'lr': opt.d_lr, 'betas': (opt.d_beta1, 0.999)}
            elif opt.optim_type == 'rmsprop':
                optim = torch.optim.RMSprop
                G_optim_kwargs = {'lr': opt.g_lr, 'alpha': opt.alpha}
                D_optim_kwargs = {'lr': opt.d_lr, 'alpha': opt.alpha}

            self.optimizer_G = optim(filter(lambda p: p.requires_grad, self.netG.parameters()), **G_optim_kwargs)
            self.optimizers.append(self.optimizer_G)

            self.optimizer_D = optim(filter(lambda p: p.requires_grad, self.netD.parameters()), **D_optim_kwargs)
            self.optimizers.append(self.optimizer_D)
           
            
            # Just a linear decay over the last 100 iterations, by default
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            if self.opt.num_discrims > 0:
                networks.print_network(self.netD)
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

        self.batch_size = input_A.shape[0]

        self.mask_size = input['mask_size'].numpy()[0]
        self.div_thresh = input['DIV_thresh']
        self.div_min = input['DIV_min']
        self.div_max = input['DIV_max']

        if self.isTrain and self.opt.num_discrims > 0:
            if self.opt.local_critic:
                assert (self.mask_size, self.mask_size) == self.critic_im_size, "Fix im dimensions in critic {} -> {}".format(self.critic_im_size, (self.mask_size, self.mask_size))
            else:
                assert input_A.shape[2:] == self.critic_im_size, "Fix im dimensions in critic {} -> {}".format(self.critic_im_size, input_A.shape[2:])


    def forward(self):
        # Thresholded, one-hot divergence map with chunk missing
        self.real_A_discrete = torch.autograd.Variable(self.input_A)
        # Complete thresholded, one-hot divergence map
        self.real_B_discrete = torch.autograd.Variable(self.input_B)
        self.real_B_fg = torch.max(self.real_B_discrete[:, [0, 2], :, :], dim=1)[0].unsqueeze(1)

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
        self.G_input = self.real_A_discrete 

        if self.opt.continent_data:
            self.G_input = torch.cat((self.G_input, self.continents.float()), dim=1)


        self.G_out = self.netG(self.G_input)
        self.fake_B_DIV = self.G_out[:, 0, :, :].unsqueeze(1)
        self.fg_classes = self.G_out[:, 1:, :, :]
        self.fg_prediction = (self.fg_classes.max(dim=1)[1] == 0).unsqueeze(1)

 
        if self.opt.grad_loss:
            self.real_B_DIV_grad_x = self.sobel_layer_x(self.real_B_DIV)
            self.real_B_DIV_grad_y = self.sobel_layer_y(self.real_B_DIV)

            self.fake_B_DIV_grad_x = self.sobel_layer_x(self.fake_B_DIV)
            self.fake_B_DIV_grad_y = self.sobel_layer_y(self.fake_B_DIV)

        scaled_thresh = self.div_thresh.repeat(1, 3) / torch.cat(
            (self.div_max, torch.ones(self.div_max.shape), -self.div_min),
            dim=1)
        scaled_thresh = scaled_thresh.view(self.fake_B_DIV.shape[0], 3, 1, 1)
        scaled_thresh = scaled_thresh.cuda() if len(self.gpu_ids) > 0 else scaled_thresh
        self.fake_B_discrete = (torch.cat(
            (-self.fake_B_DIV, torch.zeros(self.fake_B_DIV.shape, device=self.fake_B_DIV.device.type), self.fake_B_DIV)
            , dim=1) > scaled_thresh)
        plate = 1 - torch.max(self.fake_B_discrete, dim=1)[0]

        self.fake_B_discrete[:, 1, :, :].copy_(plate)

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
        

        self.fake_B_DIV_ROI = self.fake_B_DIV.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)

        self.real_B_DIV_ROI = self.real_B_DIV.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)

        self.real_B_discrete_ROI = self.real_B_discrete.masked_select(loss_mask.repeat(1, 3, 1, 1)).view(self.batch_size, 3, *im_dims)
        self.fake_B_discrete_ROI = self.fake_B_discrete.masked_select(loss_mask.repeat(1, 3, 1, 1)).view(self.batch_size, 3, *im_dims)

        self.real_B_fg_ROI = self.real_B_fg.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)
        self.fg_classes_ROI = self.fg_classes.masked_select(loss_mask.repeat(1, 2, 1, 1)).view(self.batch_size, 2, *im_dims)
        self.fg_prediction_ROI = self.fg_prediction.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)

        self.weight_mask = util.create_weight_mask(self.real_B_discrete_ROI, self.fake_B_discrete_ROI.float(), self.opt.diff_in_numerator, method='freq')
        
        if self.opt.grad_loss:
            self.real_B_DIV_grad_x = self.real_B_DIV_grad_x.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)
            self.real_B_DIV_grad_y = self.real_B_DIV_grad_y.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)

            self.fake_B_DIV_grad_x = self.fake_B_DIV_grad_x.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)
            self.fake_B_DIV_grad_y = self.fake_B_DIV_grad_y.masked_select(loss_mask).view(self.batch_size, 1, *im_dims)


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
        # self.G_input = torch.cat((self.real_A_discrete, self.mask.float()), dim=1)
        self.G_input = self.real_A_discrete

        if self.opt.continent_data:
            self.G_input = torch.cat((self.G_input, self.continents.float()), dim=1)
        
        self.fake_B_DIV = self.netG(self.G_input)

        scaled_thresh = self.div_thresh.repeat(1, 3) / torch.cat((self.div_max, torch.ones(self.div_max.shape), -self.div_min), dim=1)
        scaled_thresh = scaled_thresh.view(self.fake_B_DIV.shape[0], 3, 1, 1)
        self.fake_B_discrete = (torch.cat((-self.fake_B_DIV, torch.zeros(self.fake_B_DIV.shape), self.fake_B_DIV), dim=1) > scaled_thresh)
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


    def backward_G(self):
        self.loss_G_GAN = 0
        self.loss_G_L2 = 0

        if self.opt.num_discrims > 0:
            # Conditional data (input with chunk missing + mask) + fake data
            # Remember self.fake_B_discrete is the generator output
            if self.opt.local_critic:
                fake_AB = self.fake_B_DIV_ROI
            else:
                fake_AB = self.fake_B_DIV

            if not self.opt.no_mask_to_critic:
                fake_AB = torch.cat((fake_AB, self.mask.float()), dim=1)
            
            # Mean across batch, then across discriminators
            # We only optimise with respect to the fake prediction because
            # the first term (i.e. the real one) is independent of the generator i.e. it is just a constant term

            for p in self.netD.parameters():
                p.requires_grad = False
            
            if self.opt.use_hinge:
                self.loss_G_GAN1 = -self.netD(fake_AB)
            else:
                self.loss_G_GAN1 = self.criterionGAN(self.netD(fake_AB), True)
 
            # Trying to incentivise making this big, so it's mistaken for real
            self.loss_G_GAN = self.loss_G_GAN1

            for p in self.netD.parameters():
                p.requires_grad = True


        self.loss_G_L2_DIV_weighted = (self.weight_mask.detach() * self.criterionL2(self.fake_B_DIV_ROI, self.real_B_DIV_ROI))
        self.loss_G_L2_DIV = self.loss_G_L2_DIV_weighted.sum(dim=2).sum(dim=2) * self.opt.lambda_A

        self.loss_G_L2 += self.loss_G_L2_DIV

               # self.fake_B_DIV_ROI = self.fake_B_DIV.masked_select(self.mask.byte()).view(self.batch_size, 1, self.mask_size, self.mask_size)
        # self.real_B_DIV_ROI = self.real_B_DIV.masked_select(self.mask.byte()).view(self.batch_size, 1, self.mask_size, self.mask_size)
        
        if self.opt.grad_loss:
            grad_x_L2_img =  self.criterionL2(self.fake_B_DIV_grad_x, self.real_B_DIV_grad_x.detach())
            grad_y_L2_img =  self.criterionL2(self.fake_B_DIV_grad_y, self.real_B_DIV_grad_y.detach())

            if self.opt.weighted_grad:
                grad_x_L2_img = self.weight_mask.detach() * grad_x_L2_img
                grad_y_L2_img = self.weight_mask.detach() * grad_y_L2_img

            self.loss_L2_DIV_grad_x = (grad_x_L2_img).sum(dim=2).sum(dim=2)
            self.loss_L2_DIV_grad_y = (grad_y_L2_img).sum(dim=2).sum(dim=2)

            self.loss_G_L2 += self.loss_L2_DIV_grad_x
            self.loss_G_L2 += self.loss_L2_DIV_grad_y

        self.ce_weight_mask = util.create_weight_mask(self.real_B_fg_ROI, self.fg_prediction_ROI.float())
        
        self.loss_fg_CE_im = self.criterionCE(self.fg_classes_ROI, self.real_B_fg_ROI.long().squeeze()).unsqueeze(1) * self.ce_weight_mask.detach()
        self.loss_fg_CE = self.loss_fg_CE_im.sum(3).sum(2) * self.opt.lambda_B
        #print(self.loss_fg_CE.shape)
        #print(self.fg_prediction_ROI.shape)
        #print(self.real_B_fg_ROI.shape)
        #print(self.loss_G_L2)

        self.loss_G = self.loss_G_GAN + self.loss_G_L2 + self.loss_fg_CE

        self.loss_G = self.loss_G.mean()
        self.loss_G.backward()


    def optimize_D(self):
        if self.opt.num_discrims > 0:
            cond_data = torch.cat((self.real_A_discrete, self.mask.float()), dim=1)

            if self.opt.local_critic:
                fake_AB = self.fake_B_DIV_ROI
                real_AB = self.real_B_DIV_ROI
            else:
                fake_AB = self.fake_B_DIV
                real_AB = self.real_B_DIV
            
            if not self.opt.no_mask_to_critic:
                fake_AB = torch.cat((fake_AB, self.mask.float()), dim=1)
                real_AB = torch.cat((real_AB, self.mask.float()), dim=1)

            # stop backprop to the generator by detaching fake_B
            self.loss_D_fake = self.criterionGAN(self.netD(fake_AB.detach()), False)

            # Real
            self.loss_D_real = self.criterionGAN(self.netD(real_AB), True)

            loss = self.loss_D_fake + self.loss_D_real

            if self.opt.which_model_netD == 'wgan-gp' or self.opt.which_model_netD == 'self-attn':
                if not self.opt.use_hinge:
                    self.grad_pen_loss = self.calc_gradient_penalty(self.netD, real_AB.data, fake_AB.data) * self.opt.lambda_C
                    loss += self.grad_pen_loss

            loss = loss.mean()
            loss.backward()

            if not self.D_has_run:
                self.D_has_run = True


    def optimize_G(self):
        self.backward_G()


    def zero_optimisers(self):
        for optimiser in self.optimizers:
            optimiser.zero_grad()


    def step_optimisers(self):
        for optimiser in self.optimizers:
            optimiser.step()
                

    def get_current_errors(self):
        errors = [
            ('G', self.loss_G.data[0]),
            ('G_L2', self.loss_G_L2.data[0]),
            ('G_L2_DIV', self.loss_G_L2_DIV.data[0]),
            ('G_fg_CE', self.loss_fg_CE.data[0])]

        if self.opt.grad_loss:
            errors += [
                ('G_L2_grad_x', self.loss_L2_DIV_grad_x.data[0]),
                ('G_L2_grad_y', self.loss_L2_DIV_grad_y.data[0])
                ]

        if self.opt.num_discrims  > 0 and self.D_has_run:
            errors += [
                ('G_GAN_D', self.loss_G_GAN.data[0]),
                ('D_real', self.loss_D_real.data[0]),
                ('D_fake', self.loss_D_fake.data[0])
            ]
            if self.opt.which_model_netD == 'wgan-gp' or self.opt.which_model_netD == 'self-attn':
                if not self.opt.use_hinge:
                    errors += [('G_grad_pen', self.grad_pen_loss.data[0])]


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

        fg_prediction = util.tensor2im(self.fg_prediction.data)
        fg_prediction[mask_edge_coords] = np.max(fg_prediction)
        visuals.append(('fg_prediction', fg_prediction))

        real_B_fg = util.tensor2im(self.real_B_fg.data)
        real_B_fg[mask_edge_coords] = np.max(real_B_fg)
        visuals.append(('real_foreground', real_B_fg))

        
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
            l2_weight_mask = util.tensor2im(self.weight_mask.data)
            if not self.opt.local_loss:
                l2_weight_mask[mask_edge_coords] = np.max(l2_weight_mask)
            visuals.append(('L2 weight mask', l2_weight_mask))
            
            ce_weight_mask = util.tensor2im(self.ce_weight_mask.data)
            if not self.opt.local_loss:
                ce_weight_mask[mask_edge_coords] = np.max(ce_weight_mask)
            visuals.append(('CE weight mask', ce_weight_mask))
            
            weighted_DIV = util.tensor2im(self.loss_G_L2_DIV_weighted.data)
            if not self.opt.local_loss:
                weighted_DIV[mask_edge_coords] = np.max(weighted_DIV)
            visuals.append(('weighted_L2_loss', weighted_DIV))
            
            weighted_CE = util.tensor2im(self.loss_fg_CE_im.data)
            if not self.opt.local_loss:
                weighted_CE[mask_edge_coords] = np.max(weighted_CE)
            visuals.append(('weighted_CE_loss', weighted_CE))

        if self.opt.continent_data:
            continents = util.tensor2im(self.continents.data)
            continents[mask_edge_coords] = np.max(continents)
            visuals.append(('continents', continents))

        return OrderedDict(visuals)


    def get_current_metrics(self):
        # import skimage.io as io
        # import matplotlib.pyplot as plt
        real_DIV = self.real_B_DIV.data.numpy().squeeze()
        fake_DIV = self.fake_B_DIV.data.numpy().squeeze()

        real_DIV_local = real_DIV[np.where(self.mask.numpy().squeeze())]
        fake_DIV_local = fake_DIV[np.where(self.mask.numpy().squeeze())]

        L2_error = np.mean((real_DIV - fake_DIV)**2)
        L2_local_error = np.mean((real_DIV_local - fake_DIV_local)**2)

        metrics = [('L2_global', L2_error)]
        metrics.append(('L2_local', L2_local_error))

        return OrderedDict(metrics)


    def accumulate_metrics(self, metrics):
        # all_L2 = [metric['L2'] for metric in metrics]

        # metrics = ('L2', np.mean(all_L2))
        metrics = []

        return OrderedDict()


    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        
        self.save_network(self.netD, 'D', label, self.gpu_ids)


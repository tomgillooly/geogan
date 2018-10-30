import torch
import torchvision.transforms as transforms
import torch.autograd as autograd
from collections import OrderedDict
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

from data.geo_pickler import GeoPickler

import torch.nn.functional as F
import numpy as np
import re

from scipy.spatial.distance import directed_hausdorff, euclidean
from skimage.filters import roberts

from metrics.emd import get_emd, visualise_emd

import sys


# GAN criterion functions are optionally applied to the results from the discriminator
def wgan_criterionGAN(loss, real_label):
    return loss * (-1 if real_label else 1)


def identity(x):
    return x


class Simplified(BaseModel):
    def name(self):
        return 'SimplifiedModel'


    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        self.D_has_run = False

        # Dummy pickler that we use to create discrete images from divergence
        self.p = GeoPickler('')

        # load/define networks
        # Input channels = 3 channels for input one-hot map + mask (optional) + continents (optional)
        input_channels = opt.input_nc

        if self.opt.mask_to_G:
            input_channels += 1

        if self.opt.continent_data:
            input_channels += 1

        G_output_channels = opt.output_nc

        if opt.with_BCE:
            G_output_channels += 1


        self.netG = networks.define_G(input_channels, G_output_channels, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)


        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            
            # Inputs: 1 channel of divergence output data + mask (optional)
            discrim_input_channels = opt.output_nc

            if not opt.no_mask_to_critic:
                discrim_input_channels += 1


            # If we are only looking at the missing region
            if opt.local_critic:
                self.critic_im_size = (64, 64)
            else:
                self.critic_im_size = (256, opt.x_size)

            if self.opt.continent_data:
                discrim_input_channels += 1

            # Create discriminator
            self.netD = networks.define_D(discrim_input_channels, opt.ndf, opt.which_model_netD, opt.n_layers_D, 
                opt.norm, use_sigmoid, opt.init_type, self.gpu_ids, critic_im_size=self.critic_im_size)
            

            if len(self.gpu_ids) > 0:
                self.netD.cuda()

        if not self.isTrain or opt.continue_train or opt.restart_G:
            self.load_network(self.netG, 'G', opt.which_epoch)

            if self.isTrain and not opt.restart_G:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.opt.local_loss:
            self.im_dims = self.opt.mask_size, self.opt.mask_size
        else:
            self.im_dims = (256, self.opt.x_size)

        if self.isTrain:
            # define loss functions
            if self.opt.int_vars:
                self.criterionR = torch.nn.MSELoss(size_average=True, reduce=(not self.opt.weighted_reconstruction))
            else:
                ce_fun = torch.nn.CrossEntropyLoss(size_average=True, reduce=(not self.opt.weighted_reconstruction))
                self.criterionR = lambda test, target: ce_fun(test.view(self.opt.batchSize, self.opt.output_nc, -1),
                    target.max(dim=1)[1].view(self.opt.batchSize, -1).long())

            self.criterionBCE = torch.nn.BCELoss(size_average=True, reduce=(not self.opt.weighted_CE))

            # Choose post-processing function for reconstruction losses
            if self.opt.log_L2:
                self.processL2 = torch.log
            else:
                self.processL2 = identity

            if self.opt.log_BCE:
                self.processBCE = torch.log
            else:
                self.processBCE = identity

            # Choose post-processing function for discriminator output
            if self.opt.use_hinge:
                self.criterionGAN = hinge_criterionGAN
            elif self.opt.which_model_netD == 'wgan-gp' or self.opt.which_model_netD == 'self-attn' or self.opt.which_model_netD == 'spec-norm':
                print('using wgan criterion')
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
            self.A_path = ['serie_{}_{:05}'.format(input['folder_name'][0], input['series_number'][0])]
            
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

        if self.opt.local_loss:
            self.loss_mask = self.mask.byte()
        else:
            # if we aren't taking local loss, use entire image
            loss_mask = torch.ones(self.mask.shape).byte()
            loss_mask = loss_mask.cuda() if len(self.gpu_ids) > 0 else loss_mask
            self.loss_mask = torch.autograd.Variable(loss_mask)
       
 
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

        self.G_out = self.netG(self.G_input)
        self.fake_B_out = self.G_out[:, :self.opt.output_nc, :, :]
        self.fake_B_out_ROI = self.fake_B_out.masked_select(self.loss_mask).view(self.batch_size, self.fake_B_out.shape[1], *self.im_dims)

        # If we're creating the foreground image, just use that as discrete
        self.fake_B_fg = torch.nn.Sigmoid()(self.G_out[:, -1, :, :].unsqueeze(1))
        self.fake_fg_discrete = self.fake_B_fg > 0.5

        self.fake_B_DIV = self.fake_B_out
        self.fake_B_DIV_ROI = self.fake_B_out_ROI

        self.real_B_fg_ROI = self.real_B_fg.masked_select(self.loss_mask).view(self.batch_size, 1, *self.im_dims)
        self.fake_fg_discrete_ROI = self.fake_fg_discrete.masked_select(self.loss_mask).view(self.batch_size, 1, *self.im_dims)
        self.fake_B_fg_ROI = self.fake_B_fg.masked_select(self.loss_mask).view(self.batch_size, 1, *self.im_dims)

        self.real_B_DIV_ROI = self.real_B_DIV.masked_select(self.loss_mask).view(self.batch_size, 1, *self.im_dims)
        
        self.fake_B_discrete_ROI = self.fake_B_discrete.masked_select(self.loss_mask.repeat(1, 3, 1, 1)).view(self.batch_size, 3, *self.im_dims)
        self.real_B_discrete_ROI = self.real_B_discrete.masked_select(self.loss_mask.repeat(1, 3, 1, 1)).view(self.batch_size, 3, *self.im_dims)

        self.real_B_out_ROI = self.real_B_DIV_ROI


    # no backprop gradients
    def test(self):
        self.real_A_discrete = torch.autograd.Variable(self.input_A, volatile=True)
        self.real_B_discrete = torch.autograd.Variable(self.input_B, volatile=True)
        self.real_B_fg = torch.max(self.real_B_discrete[:, [0, 2], :, :], dim=1)[0].unsqueeze(1)

        self.real_A_DIV = torch.autograd.Variable(self.input_A_DIV)
        self.real_B_DIV = torch.autograd.Variable(self.input_B_DIV)

        self.mask = torch.autograd.Variable(self.mask)
        
        if self.opt.continent_data:
            self.continents = torch.autograd.Variable(self.continent_img)

        # mask_var = Variable(self.mask.float(), volatile=True)
        # self.G_input = torch.cat((self.real_A_discrete, self.mask.float()), dim=1)
        self.G_input = self.real_A_discrete

        if self.opt.mask_to_G:
            self.G_input = torch.cat((self.G_input, self.mask.float()), dim=1)

        if self.opt.continent_data:
            self.G_input = torch.cat((self.G_input, self.continents.float()), dim=1)
        
        self.G_out = self.netG(self.G_input)
        self.fake_B_out = self.G_out[:, :self.opt.output_nc, :, :]
        
        if self.opt.with_BCE:
            self.fake_B_fg = torch.nn.Sigmoid()(self.G_out[:, -1:, :, :])
            self.fake_fg_discrete = self.fake_B_fg > 0.5


        # Work out the threshold from quantification factor
        # tmp_dict = {'A_DIV': self.fake_B_DIV.data[0].numpy().squeeze()}
        # self.p.create_one_hot(tmp_dict, 0.5)
        # self.fake_B_discrete_05 = tmp_dict['A']
        # self.p.create_one_hot(tmp_dict, 0.2)
        # self.fake_B_discrete_02 = tmp_dict['A']
        # self.p.create_one_hot(tmp_dict, 0.1)

        
        self.real_B_DIV_ROI = self.real_B_DIV.masked_select(self.loss_mask).view(self.batch_size, 1, *self.im_dims)

        self.real_B_discrete_ROI = self.real_B_discrete.masked_select(self.loss_mask.repeat(1, 3, 1, 1)).view(self.batch_size, 3, *self.im_dims)
        self.fake_B_discrete_ROI = self.fake_B_discrete.masked_select(self.loss_mask.repeat(1, 3, 1, 1)).view(self.batch_size, 3, *self.im_dims)

        self.real_B_out_ROI = self.real_B_DIV_ROI

        self.real_B_fg_ROI = self.real_B_fg.masked_select(self.loss_mask).view(self.batch_size, 1, *self.im_dims)
        self.fake_B_fg_ROI = self.fake_B_fg.masked_select(self.loss_mask).view(self.batch_size, 1, *self.im_dims)
        self.fake_fg_discrete_ROI = self.fake_fg_discrete.masked_select(self.loss_mask).view(self.batch_size, 1, *self.im_dims)

        
        # self.fake_B_discrete_01 = tmp_dict['A']


    # get image paths
    def get_image_paths(self):
        return self.A_path


    def calc_gradient_penalty(self, netD, real_data, fake_data):
        # Calculate gradient penalty of points interpolated between real and fake pairs
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

        # Conditional data (input with chunk missing + mask) + fake data
        # Remember self.fake_B_discrete is the generator output
        fake_AB = self.fake_B_DIV

        fake_AB = torch.cat((fake_AB, self.mask.float()), dim=1)
        
        # Mean across batch, then across discriminators
        # We only optimise with respect to the fake prediction because
        # the first term (i.e. the real one) is independent of the generator i.e. it is just a constant term

        for p in self.netD.parameters():
            p.requires_grad = False
        
        self.loss_G_GAN1 = self.criterionGAN(self.netD(fake_AB), True)

        # Trying to incentivise making this big, so it's mistaken for real
        self.loss_G_GAN = self.loss_G_GAN1 * self.opt.lambda_D

        for p in self.netD.parameters():
            p.requires_grad = True


        ##### L2 Loss
        self.loss_G_rec = self.criterionR(self.fake_B_out_ROI, self.real_B_out_ROI)

        self.loss_G_rec = self.processL2(self.loss_G_rec * self.opt.lambda_A + 1e-8) * self.opt.lambda_A2

        # self.loss_G_rec += self.loss_G_L2_rec
        
        self.loss_G = self.loss_G_GAN + self.loss_G_rec


        ##### BCE Loss
        self.loss_fg_CE = self.criterionBCE(self.fake_B_fg_ROI, self.real_B_fg_ROI.float())

        self.loss_fg_CE = (self.ce_weight_mask.detach() * self.loss_fg_CE).sum(3).sum(2)

        self.loss_fg_CE = self.processBCE(self.loss_fg_CE * self.opt.lambda_B + 1e-8) * self.opt.lambda_B2

        self.loss_G += self.loss_fg_CE


        self.loss_G = self.loss_G.mean()
        self.loss_G.backward()


    def optimize_D(self):
        cond_data = torch.cat((self.real_A_discrete, self.mask.float()), dim=1)

        fake_AB = self.fake_B_DIV
        real_AB = self.real_B_DIV
        
        fake_AB = torch.cat((fake_AB, self.mask.float()), dim=1)
        real_AB = torch.cat((real_AB, self.mask.float()), dim=1)

        # stop backprop to the generator by detaching fake_B
        self.loss_D_fake = self.criterionGAN(self.netD(fake_AB.detach()), False)

        # Real
        self.loss_D_real = self.criterionGAN(self.netD(real_AB), True)

        loss = self.loss_D_fake + self.loss_D_real

        # self.grad_pen_loss = self.calc_gradient_penalty(self.netD, real_AB.data, fake_AB.data) * self.opt.lambda_C
        # loss += self.grad_pen_loss

        loss = loss.mean()
        loss.backward()


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
            ('G_rec', self.loss_G_rec.data[0])
        ]

        if self.opt.grad_loss:
            errors += [
                ('G_L2_grad_x', self.loss_L2_DIV_grad_x.data[0]),
                ('G_L2_grad_y', self.loss_L2_DIV_grad_y.data[0])
                ]

        if self.opt.with_BCE:
            errors += [
                ('G_fg_CE', self.loss_fg_CE.data[0])
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

        # fake_B_discrete = util.tensor2im(self.fake_B_discrete.data)
        # fake_B_discrete[mask_edge_coords] = np.max(fake_B_discrete)
        # visuals.append(('output_one_hot', fake_B_discrete))

        real_A_DIV = util.tensor2im(self.real_A_DIV.data)
        real_A_DIV[mask_edge_coords] = np.max(real_A_DIV)
        visuals.append(('input_divergence', real_A_DIV))

        real_B_DIV = util.tensor2im(self.real_B_DIV.data)
        real_B_DIV[mask_edge_coords] = np.max(real_B_DIV)
        visuals.append(('ground_truth_divergence', real_B_DIV))

        fake_B_DIV = util.tensor2im(self.fake_B_DIV.data)
        fake_B_DIV[mask_edge_coords] = np.max(fake_B_DIV)
        visuals.append(('output_divergence', fake_B_DIV))
    
        fake_B_fg = util.tensor2im(self.fake_B_fg.data)
        fake_B_fg[mask_edge_coords] = np.max(fake_B_fg)
        visuals.append(('fake_B_fg', fake_B_fg))

        fake_fg_discrete = util.tensor2im(self.fake_fg_discrete.data.float())
        fake_fg_discrete[mask_edge_coords] = np.max(fake_fg_discrete)
        visuals.append(('fake_fg_discrete', fake_fg_discrete))

        real_B_fg = util.tensor2im(self.real_B_fg.data)
        real_B_fg[mask_edge_coords] = np.max(real_B_fg)
        visuals.append(('real_foreground', real_B_fg))

        if not self.isTrain:
            visuals.append(('emd_ridge_error', self.emd_ridge_error))
            visuals.append(('emd_subduction_error', self.emd_subduction_error))

        return OrderedDict(visuals)


    def get_current_metrics(self):
        from collections import defaultdict

        real_disc_local = self.real_B_discrete.masked_select(self.mask.repeat(1, 3, 1, 1)).view(1, 3, self.mask_size, self.mask_size).data.numpy().squeeze().transpose(1, 2, 0)
        metrics = []
        if self.opt.int_vars:
            # import skimage.io as io
            # import matplotlib.pyplot as plt
            real_DIV = self.real_B_DIV.data.numpy().squeeze()
            real_disc = self.real_B_discrete.data.numpy().squeeze().transpose(1, 2, 0)
            fake_DIV = self.fake_B_DIV.data.numpy().squeeze()

            real_DIV_local = self.real_B_DIV.masked_select(self.mask).view(1, 1, self.mask_size, self.mask_size).numpy().squeeze()
            fake_DIV_local = self.fake_B_DIV.masked_select(self.mask).view(1, 1, self.mask_size, self.mask_size).data.numpy().squeeze()

            L2_error = np.mean((real_DIV - fake_DIV)**2)
            L2_local_error = np.mean((real_DIV_local - fake_DIV_local)**2)

            metrics.append(('L2_global', L2_error))
            metrics.append(('L2_local', L2_local_error))

            low_thresh = 2e-4
            high_thresh = max(np.max(fake_DIV_local), np.abs(np.min(fake_DIV_local)))
            
            # Somehow goofed and produced inverted divergence maps for circles/ellipses, so we sometimes need to flip to compare
            tmp = {'A_DIV': fake_DIV_local * (-1 if self.opt.invert_ridge else 1)}
            #print(np.max(tmp['A_DIV']), np.min(tmp['A_DIV']))

            scores = np.ones((5, 1)) * np.inf

            results_cache = defaultdict(dict)
            print('search_iter: ')
            for search_iter in range(10):
                print('{}... '.format(search_iter), end='\r')

                # Threshold at equal intervals between low and high, find best score
                thresholds = np.linspace(low_thresh, high_thresh, 5)

                for thresh_idx, thresh in enumerate(thresholds):
                    if scores[thresh_idx] != np.inf:
                        continue

                    self.p.create_one_hot(tmp, thresh, skel=self.opt.skel_metric)
                    tmp_disc = tmp['A']

                    s = []
                    for i in [0, 2]:
                        tmp_emd, pairs = get_emd(tmp_disc[:,:,i], real_disc_local[:,:,i], average=True, return_pairs=True)

                        s.append(tmp_emd)
                        results_cache[thresh][i] = {'pairs':  pairs}
                        results_cache[thresh][i]['score'] = tmp_emd
                    scores[thresh_idx] = (np.mean(s))
                
                best_idx = np.argmin(scores)
                DIV_thresh = thresholds[best_idx]
                best_score = scores.ravel()[best_idx]
               
                high_idx = best_idx + 1
                low_idx = best_idx - 1

                if high_idx >= len(thresholds):
                    high_idx -= 1

                if low_idx < 0:
                    low_idx += 1

                high_thresh = thresholds[high_idx]
                low_thresh = thresholds[low_idx]

                print(scores.ravel()[best_idx])
                scores[0] = scores[low_idx]
                scores[-1] = scores[high_idx]
                scores[1:-1] = np.inf

            print('Best thresh/score : {}/{}'.format(DIV_thresh, best_score))
            self.p.create_one_hot(tmp, DIV_thresh, skel=self.opt.skel_metric)
            print('Created new one-hot')

            print('Computing emd 0 ', end='')
            # emd_cost0, im0 = get_emd(tmp['A'][:, :, 0], real_disc_local[:, :, 0], visualise=True)
            results = results_cache[DIV_thresh][0]
            emd_cost0 = results['score']
            im0 = visualise_emd(emd_cost0, *self.im_dims, **results['pairs'])

            print('Computing emd 1 ', end='')
            # emd_cost1, im1 = get_emd(tmp['A'][:, :, 2], real_disc_local[:, :, 2], visualise=True)
            results = results_cache[DIV_thresh][2]
            emd_cost1 = results['score']
            im1 = visualise_emd(emd_cost1, *self.im_dims, **results['pairs'])

            tmp['A_DIV'] = fake_DIV * (-1 if self.opt.invert_ridge else 1)
            print('Creating full one hot image')
            self.p.create_one_hot(tmp, DIV_thresh, skel=self.opt.skel_metric)
            self.fake_B_discrete.data.copy_(torch.from_numpy(tmp['A'].transpose(2, 0, 1)))
            self.emd_ridge_error = im0
            self.emd_subduction_error = im1
        else:
            self.fake_disc_local = self.fake_B_discrete.masked_select(self.mask.repeat(1, 3, 1, 1)).view(1, 3, self.mask_size, self.mask_size).data.numpy().squeeze().transpose(1, 2, 0)
        
            print('Computing emd 0 ', end='')
            emd_cost0, results = get_emd(self.fake_disc_local[:, :, 0], real_disc_local[:, :, 0], return_pairs=True)
            im0 = visualise_emd(emd_cost0, *self.im_dims, **results)
            print('Computing emd 1 ', end='')
            emd_cost1, results = get_emd(self.fake_disc_local[:, :, 2], real_disc_local[:, :, 2], return_pairs=True)
            im1 = visualise_emd(emd_cost1, *self.im_dims, **results)

            self.emd_ridge_error = im0
            self.emd_subduction_error = im1


        metrics += [('EMD_ridge', emd_cost0), ('EMD_subduction', emd_cost1), ('EMD_mean', (emd_cost0+emd_cost1)/2)]
        print('Done')
        return OrderedDict(metrics)


    def accumulate_metrics(self, metrics):
        a_metrics = []

        if self.opt.int_vars:
            a_metrics.append(('L2_global', np.mean([metric['L2_global'] for metric in metrics])))
            a_metrics.append(('L2_local', np.mean([metric['L2_local'] for metric in metrics])))

        a_metrics.append(('EMD_ridge', np.mean([metric['EMD_ridge'] for metric in metrics])))
        a_metrics.append(('EMD_subduction', np.mean([metric['EMD_subduction'] for metric in metrics])))
        a_metrics.append(('EMD_mean', np.mean([metric['EMD_mean'] for metric in metrics])))

        return OrderedDict(a_metrics)


    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        
        self.save_network(self.netD, 'D', label, self.gpu_ids)


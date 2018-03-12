import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import skimage.io as io
import sys

class Pix2PixGeoModel(BaseModel):
    def name(self):
        return 'Pix2PixGeoModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        self.netG = networks.define_G(opt.input_nc + 1, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        # print(self.netG)
        # self.netG = nn.Sequential(self.netG, nn.Softmax2d())

        self.netG_cont = nn.Sequential(
            nn.Conv2d(in_channels=opt.output_nc, out_channels=1, kernel_size=1),
            )
        # self.netG_disc = nn.Sequential(self.netG, nn.LogSoftmax(dim=3))

        if len(self.gpu_ids) > 0:
            self.netG_cont.cuda(self.gpu_ids[0])
            # self.netG_disc.cuda(self.gpu_ids[0])

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD1 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD2 = networks.define_D(opt.input_nc + 1, opt.ndf,
                                          opt.which_model_netD,
                                          opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch)

        if self.isTrain:
            # Image pool not doing anything in this model because size is set to zero, just
            # returns input as Variable
            self.fake_AB_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL2 = torch.nn.MSELoss(size_average=True)
            self.criterionCE = torch.nn.NLLLoss2d

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_cont = torch.optim.Adam(self.netG_cont.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D1 = torch.optim.Adam(self.netD1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D2 = torch.optim.Adam(self.netD2.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_G_cont)
            self.optimizers.append(self.optimizer_D1)
            self.optimizers.append(self.optimizer_D2)
            # Just a linear decay over the last 100 iterations, by default
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD1)
        print('-----------------------------------------------')

    def set_input(self, input):
        # This model is B to A by default
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_A_cont = input['A_cont' if AtoB else 'B_cont']
        input_B_cont = input['B_cont' if AtoB else 'A_cont']
        mask = input['mask']
        
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
            input_A_cont = input_A_cont.cuda(self.gpu_ids[0], async=True)
            input_B_cont = input_B_cont.cuda(self.gpu_ids[0], async=True)
            mask = mask.cuda(self.gpu_ids[0], async=True)
        
        self.input_A = input_A
        self.input_B = input_B
        self.input_A_cont = input_A_cont
        self.input_B_cont = input_B_cont
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        self.mask = mask

    def forward(self):
        # Continuous divergence map with chunk missing
        self.real_A_cont = Variable(self.input_A_cont)#, requires_grad=False)
        # Thresholded divergence map with chunk missing
        self.real_A_discrete = Variable(self.input_A)#, requires_grad=False)
        # Entire thresholded divergence map
        self.real_B_discrete = Variable(self.input_B)#, requires_grad=False)
        # Entire continuous divergence map
        self.real_B_cont = Variable(self.input_B_cont)#, requires_grad=False)
        
        # Produces three channel output with class probability assignments
        self.fake_B_discrete = self.netG(torch.cat((self.real_A_discrete, Variable(self.mask.float())), dim=1))
        # Create continuous divergence field from class probabilities
        self.fake_B_cont = self.netG_cont(self.fake_B_discrete)
        # Find log probabilities for NLL step in backprop
        
        self.fake_B_classes = F.log_softmax(self.fake_B_discrete, dim=1)
        # Apply max and normalise to get -1 to 1 range
        self.fake_B_classes = torch.max(self.fake_B_classes, dim=1)

        self.real_B_classes = F.log_softmax(self.real_B_discrete, dim=1)
        self.real_B_classes = torch.max(self.real_B_discrete, dim=1, keepdim=False)[1]

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D1(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # In this case real_A, the input, is our conditional vector
        fake_AB = torch.cat((self.real_A_discrete, self.fake_B_discrete), 1)
        pred_fake = self.netD1(fake_AB.detach())
        self.loss_D1_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A_discrete, self.real_B_discrete), 1)
        pred_real = self.netD1(real_AB)
        self.loss_D1_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D1 = (self.loss_D1_fake + self.loss_D1_real) * 0.5

        self.loss_D1.backward()

    def backward_D2(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # In this case real_A, the input, is our conditional vector
        fake_AB = torch.cat((self.real_A_discrete, self.fake_B_cont), 1)
        pred_fake = self.netD2(fake_AB.detach())
        self.loss_D2_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((self.real_A_discrete, self.real_B_cont), 1)
        pred_real = self.netD2(real_AB)
        self.loss_D2_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5

        self.loss_D2.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        # Note that we don't detach here because we DO want to backpropagate
        # to the generator this time

        fake_AB = torch.cat((self.real_A_discrete, self.fake_B_discrete), 1)
        pred_fake1 = self.netD1(fake_AB)
        
        fake_AB = torch.cat((self.real_A_discrete, self.fake_B_cont), 1)
        pred_fake2 = self.netD2(fake_AB)

        # We only optimise with respect to the fake prediction because
        # the first term (i.e. the real one) is independent of the generator i.e. it is just a constant term
        self.loss_G_GAN1 = self.criterionGAN(pred_fake1, True)
        self.loss_G_GAN2 = self.criterionGAN(pred_fake2, True)

        # Second, G(A) = B
        self.loss_G_L2 = self.criterionL2(self.fake_B_cont, self.real_B_cont) * self.opt.lambda_A

        weights = torch.sum(self.fake_B_discrete.view(3, -1), dim=1)
        weights /= torch.sum(weights)

        ce_fun = self.criterionCE(weight=weights.data)

        self.loss_G_CE = ce_fun(F.log_softmax(self.fake_B_discrete, dim=1), self.real_B_classes) * self.opt.lambda_B

        self.loss_G = self.loss_G_GAN1 + self.loss_G_GAN2 + self.loss_G_L2 + self.loss_G_CE

        self.loss_G.backward()

    def optimize_parameters(self):
        # Doesn't do anything with discriminator, just populates input (conditional), 
        # target and generated data in object
        self.forward()

        # Nothing fancy, no cyclical business to worry about
        self.optimizer_D1.zero_grad()
        self.backward_D1()
        self.optimizer_D1.step()

        self.optimizer_D2.zero_grad()
        self.backward_D2()
        self.optimizer_D2.step()

        self.optimizer_G.zero_grad()
        self.optimizer_G_cont.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_G_cont.step()

    def get_current_errors(self):
        return OrderedDict([
            ('G', self.loss_G.data[0]),
            ('G_GAN_D1', self.loss_G_GAN1.data[0]),
            ('G_GAN_D2', self.loss_G_GAN2.data[0]),
            ('G_L2', self.loss_G_L2.data[0]),
            ('G_CE', self.loss_G_CE.data[0]),
            ('D1_real', self.loss_D1_real.data[0]),
            ('D1_fake', self.loss_D1_fake.data[0]),
            ('D2_real', self.loss_D2_real.data[0]),
            ('D2_fake', self.loss_D2_fake.data[0])
            ])

    def get_current_visuals(self):
        # print(np.unique(self.real_A_discrete.data))
        # print(self.fake_B_discrete.data.shape)

        fake_B_one_hot = torch.zeros(self.fake_B_discrete.shape)
        fake_B_classes = torch.max(self.fake_B_discrete, dim=1, keepdim=True)[1]

        fake_B_one_hot.scatter_(1, fake_B_classes.data.cpu(), 1.0)


        real_A_discrete = util.tensor2im(self.real_A_discrete.data)
        real_A_continuous = util.tensor2im(self.real_A_cont.data)
        real_B_discrete = util.tensor2im(self.real_B_discrete.data)
        real_B_continuous = util.tensor2im(self.real_B_cont.data)
        fake_B_discrete = util.tensor2im(self.fake_B_discrete.data)
        fake_B_one_hot = util.tensor2im(fake_B_one_hot)
        fake_B_continuous = util.tensor2im(self.fake_B_cont.data)
        return OrderedDict([
            ('real_A_discrete', real_A_discrete), ('real_A_continuous', real_A_continuous), 
            ('real_B_discrete', real_B_discrete), ('real_B_continuous', real_B_continuous), 
            ('fake_B_one_hot', fake_B_one_hot),
            ('fake_B_discrete', fake_B_discrete),
            ('fake_B_continuous', fake_B_continuous)
            ])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netG_cont, 'G_cont', label, self.gpu_ids)
        self.save_network(self.netD1, 'D1', label, self.gpu_ids)
        self.save_network(self.netD2, 'D2', label, self.gpu_ids)

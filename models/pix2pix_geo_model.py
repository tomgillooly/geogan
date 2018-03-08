import torch
from collections import OrderedDict
from torch.autograd import Variable
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

import torch.nn as nn


class Pix2PixGeoModel(BaseModel):
    def name(self):
        return 'Pix2PixGeoModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        # load/define networks
        self.netG = networks.define_G(opt.input_nc, 3, opt.ngf,
                                      opt.which_model_netG, opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)

        # print(self.netG)
        # self.netG = nn.Sequential(self.netG, nn.Softmax2d())

        self.netG_cont = nn.Conv2d(in_channels=3, out_channels=opt.output_nc, kernel_size=1)
        self.netG_disc = nn.Sequential(self.netG, nn.LogSoftmax(dim=3))

        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD = networks.define_D(opt.output_nc, opt.ndf,
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
            self.criterionL2 = torch.nn.MSELoss()
            self.criterionCE = torch.nn.NLLLoss2d()

            # initialize optimizers
            self.schedulers = []
            self.optimizers = []
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            # Just a linear decay over the last 100 iterations, by default
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD)
        print('-----------------------------------------------')

    def set_input(self, input):
        # This model is B to A by default
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_A_cont = input['A_cont' if AtoB else 'B_cont']
        input_B_cont = input['B_cont' if AtoB else 'A_cont']
        
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
            input_A_cont = input_A_cont.cuda(self.gpu_ids[0], async=True)
            input_B_cont = input_B_cont.cuda(self.gpu_ids[0], async=True)
        
        self.input_A = input_A
        self.input_B = input_B
        self.input_A_cont = input_A_cont
        self.input_B_cont = input_B_cont
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A_cont = Variable(self.input_A_cont, requires_grad=False)
        self.real_A_discrete = Variable(self.input_A, requires_grad=False)
        self.real_B_discrete = Variable(self.input_B, requires_grad=False)
        self.real_B_cont = Variable(self.input_B_cont, requires_grad=False)
        
        self.fake_B_discrete = self.netG_disc(self.real_A_cont)
        self.fake_B_cont = self.netG_cont(self.fake_B_discrete)

    # no backprop gradients
    def test(self):
        self.real_A = Variable(self.input_A, volatile=True)
        self.fake_B = self.netG(self.real_A)
        self.real_B = Variable(self.input_B, volatile=True)

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Why was there a concatenation here? I think for the cyclical thing
        pred_fake = self.netD(self.fake_B_cont.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real
        # In this case real_A, the input, is our conditional vector
        pred_real = self.netD(self.real_B_cont)
        self.loss_D_real = self.criterionGAN(pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward()

    def backward_G(self):
        # First, G(A) should fake the discriminator
        # Note that we don't detach here because we DO want to backpropagate
        # to the generator this time

        # fake_AB = torch.cat((self.real_A, self.fake_B_cont), 1)
        pred_fake = self.netD(self.fake_B_cont)

        # We only optimise with respect to the fake prediction because
        # the first term (i.e. the real one) is independent of the generator i.e. it is just a constant term
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)

        # Second, G(A) = B
        self.loss_G_L2 = self.criterionL2(self.fake_B_cont, self.real_A_cont) * self.opt.lambda_A
        
        self.loss_G_CE = self.criterionCE(self.fake_B_discrete, torch.squeeze(self.real_A_discrete, dim=3)) * self.opt.lambda_B

        self.loss_G = self.loss_G_GAN + self.loss_G_L2 + self.loss_G_CE

        self.loss_G.backward()

    def optimize_parameters(self):
        # Doesn't do anything with discriminator, just populates input (conditional), 
        # target and generated data in object
        self.forward()

        # Nothing fancy, no cyclical business to worry about
        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()

    def get_current_errors(self):
        return OrderedDict([('G_GAN', self.loss_G_GAN.data[0]),
                            ('G_L1', self.loss_G_L1.data[0]),
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0])
                            ])

    def get_current_visuals(self):
        real_A = util.tensor2im(self.real_A.data)
        fake_B = util.tensor2im(self.fake_B.data)
        real_B = util.tensor2im(self.real_B.data)
        return OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('real_B', real_B)])

    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)

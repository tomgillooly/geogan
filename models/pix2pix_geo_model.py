import torch
import torchvision.transforms as transforms
import torch.autograd as autograd
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

from scipy.spatial.distance import directed_hausdorff, euclidean
from skimage.filters import roberts

import sys

# Weight init procedure taken from  https://github.com/pytorch/examples/blob/master/dcgan/main.py#L131
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('InstanceNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class DiscriminatorWGANGP(nn.Module):

    def __init__(self, in_dim, image_dims, dim=64):
        super(DiscriminatorWGANGP, self).__init__()

        def conv_ln_lrelu(in_dim, out_dim):
            return nn.Sequential(
                nn.Conv2d(in_dim, out_dim, 5, 2, 2),
                # Since there is no effective implementation of LayerNorm,
                # we use InstanceNorm2d instead of LayerNorm here.
                # Gulrajanis code uses TensorFlow batch normalisation
                nn.InstanceNorm2d(out_dim, affine=True),
                nn.LeakyReLU(0.2))

        self.ls = nn.Sequential(
            nn.Conv2d(in_dim, dim, 5, 2, 2), nn.LeakyReLU(0.2),         # (b, c, x, y) -> (b, dim, x/2, y/2)
            conv_ln_lrelu(dim, dim * 2),                                # (b, dim, x/2, y/2) -> (b, dim*2, x/4, y/4)
            conv_ln_lrelu(dim * 2, dim * 4),                            # (b, dim*2, x/4, y/4) -> (b, dim*4, x/8, y/8)
            conv_ln_lrelu(dim * 4, dim * 8),                            # (b, dim*4, x/8, y/8) -> (b, dim*8, x/16, y/16)
            nn.Conv2d(dim * 8, 1, 
                (int(image_dims[0]/16 + 0.5), int(image_dims[1]/16 + 0.5)))) # (b, dim*8, x/16, y/16) -> (b, 1, 1, 1)

    def forward(self, x):
        y = self.ls(x)
        y = y.view(-1)
        return y

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

        self.netG_DIV = nn.Sequential(
            nn.Conv2d(in_channels=opt.output_nc, out_channels=1, kernel_size=1),
            )
        self.netG_Vx = nn.Sequential(
            nn.Conv2d(in_channels=opt.output_nc, out_channels=1, kernel_size=1),
            )
        self.netG_Vy = nn.Sequential(
            nn.Conv2d(in_channels=opt.output_nc, out_channels=1, kernel_size=1),
            )
        # self.netG_disc = nn.Sequential(self.netG, nn.LogSoftmax(dim=3))

        if len(self.gpu_ids) > 0:
            self.netG_DIV.cuda(self.gpu_ids[0])
            self.netG_Vx.cuda(self.gpu_ids[0])
            self.netG_Vy.cuda(self.gpu_ids[0])
            # self.netG_disc.cuda(self.gpu_ids[0])

        if self.isTrain:
            # use_sigmoid = opt.no_lsgan
            # use_sigmoid = True
            # self.netD1 = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf,
            #                               # opt.which_model_netD,
            #                               'wgan',
            #                               opt.n_layers_D, 'none', use_sigmoid, opt.init_type, self.gpu_ids,
            #                               n_linear=1860)
                                          # n_linear=int((512*256)/70))
            
            # Discrete input data + discrete output data
            self.netD1s = [DiscriminatorWGANGP(opt.input_nc + opt.output_nc, (256, 512), opt.ndf)
                            for _ in range(self.opt.num_discrims)]

            # Discrete input data + DIV, Vx, Vy
            self.netD2s = [DiscriminatorWGANGP(opt.input_nc + 3, (256, 512), opt.ndf)
                            for _ in range(self.opt.num_discrims)]


            [netD1.apply(weights_init) for netD1 in self.netD1s]
            [netD2.apply(weights_init) for netD2 in self.netD2s]


            if len(self.gpu_ids) > 0:
                [netD.cuda() for netD in self.netD1s]
                [netD.cuda() for netD in self.netD2s]
            
            # self.netD2 = networks.define_D(opt.input_nc + 3, opt.ndf,
            #                               # opt.which_model_netD,
            #                               'wgan',
            #                               opt.n_layers_D, 'none', use_sigmoid, opt.init_type, self.gpu_ids,
            #                               n_linear=1860)
            #                               # n_linear=int((512*256)/70))

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netG, 'G', opt.which_epoch)
            self.load_network(self.netG_DIV, 'G_DIV', opt.which_epoch)
            self.load_network(self.netG_DIV, 'G_Vx', opt.which_epoch)
            self.load_network(self.netG_DIV, 'G_Vy', opt.which_epoch)
            if self.isTrain:
                [self.load_network(netD1s[i], 'D1_%d' % i, label, opt.which_epoch) for i in range(len(self.netD1s))]
                [self.load_network(netD2s[i], 'D2_%d' % i, label, opt.which_epoch) for i in range(len(self.netD2s))]

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
            self.optimizer_G_DIV = torch.optim.Adam(self.netG_DIV.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_Vx = torch.optim.Adam(self.netG_Vx.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_G_Vy = torch.optim.Adam(self.netG_Vy.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D1s = [torch.optim.Adam(netD1.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999)) for netD1 in self.netD1s]
            self.optimizer_D2s = [torch.optim.Adam(netD2.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999)) for netD2 in self.netD2s]
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_G_DIV)
            self.optimizers.append(self.optimizer_G_Vx)
            self.optimizers.append(self.optimizer_G_Vy)
            self.optimizers += self.optimizer_D1s
            self.optimizers += self.optimizer_D2s

            # Just a linear decay over the last 100 iterations, by default
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG)
        if self.isTrain:
            networks.print_network(self.netD1s[0])
        print('-----------------------------------------------')

    def set_input(self, input):
        # This model is B to A by default
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A' if AtoB else 'B']
        input_B = input['B' if AtoB else 'A']
        input_A_DIV = input['A_DIV' if AtoB else 'B_DIV']
        input_B_DIV = input['B_DIV' if AtoB else 'A_DIV']
        input_A_Vx = input['A_Vx' if AtoB else 'B_Vx']
        input_B_Vx = input['B_Vx' if AtoB else 'A_Vx']
        input_A_Vy = input['A_Vy' if AtoB else 'B_Vy']
        input_B_Vy = input['B_Vy' if AtoB else 'A_Vy']
        mask = input['mask']
        
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
            input_A_DIV = input_A_DIV.cuda(self.gpu_ids[0], async=True)
            input_B_DIV = input_B_DIV.cuda(self.gpu_ids[0], async=True)
            input_A_Vx = input_A_Vx.cuda(self.gpu_ids[0], async=True)
            input_B_Vx = input_B_Vx.cuda(self.gpu_ids[0], async=True)
            input_A_Vy = input_A_Vy.cuda(self.gpu_ids[0], async=True)
            input_B_Vy = input_B_Vy.cuda(self.gpu_ids[0], async=True)
            mask = mask.cuda(self.gpu_ids[0], async=True)
        
        self.input_A = input_A
        self.input_B = input_B
        self.input_A_DIV = input_A_DIV
        self.input_B_DIV = input_B_DIV
        self.input_A_Vx = input_A_Vx
        self.input_B_Vx = input_B_Vx
        self.input_A_Vy = input_A_Vy
        self.input_B_Vy = input_B_Vy
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

        self.mask = mask
        mask_x1 = input['mask_x1']
        mask_x2 = input['mask_x2']
        mask_y1 = input['mask_y1']
        mask_y2 = input['mask_y2']

        self.batch_size =input_A.shape[0]
        # Masks are always the same size
        self.mask_size_y = mask_y2[0] - mask_y1[0]
        self.mask_size_x = mask_x2[0] - mask_x1[0]

    def forward(self):
        # Continuous divergence map with chunk missing
        self.real_A_DIV = Variable(self.input_A_DIV)#, requires_grad=False)
        # Vector fields with chunk missing
        self.real_A_Vx = Variable(self.input_A_Vx)#, requires_grad=False)
        self.real_A_Vy = Variable(self.input_A_Vy)#, requires_grad=False)
        # Thresholded divergence map with chunk missing
        self.real_A_discrete = Variable(self.input_A)#, requires_grad=False)
        # Entire thresholded divergence map
        self.real_B_discrete = Variable(self.input_B)#, requires_grad=False)
        # Entire continuous divergence map
        self.real_B_DIV = Variable(self.input_B_DIV)#, requires_grad=False)
        # Entire vector field
        self.real_B_Vx = Variable(self.input_B_Vx)#, requires_grad=False)
        self.real_B_Vy = Variable(self.input_B_Vy)#, requires_grad=False)

        self.mask = Variable(self.mask)
        
        # Produces three channel output with class probability assignments
        self.fake_B_discrete = self.netG(torch.cat((self.real_A_discrete, self.mask.float()), dim=1))
        # Create continuous divergence field from class probabilities
        self.fake_B_DIV = self.netG_DIV(self.fake_B_discrete)
        # Create vector field from class probabilities
        self.fake_B_Vx = self.netG_Vx(self.fake_B_discrete)
        self.fake_B_Vy = self.netG_Vy(self.fake_B_discrete)
        # Find log probabilities for NLL step in backprop
        
        self.fake_B_classes = F.log_softmax(self.fake_B_discrete, dim=1)
        # Apply max and normalise to get -1 to 1 range
        self.fake_B_classes = torch.max(self.fake_B_classes, dim=1, keepdim=True)[1]

        self.real_B_classes = F.log_softmax(self.real_B_discrete, dim=1)
        self.real_B_classes = torch.max(self.real_B_discrete, dim=1, keepdim=False)[1]

        self.fake_B_one_hot = torch.zeros(self.fake_B_discrete.shape)
        
        self.fake_B_one_hot.scatter_(1, self.fake_B_classes.data.cpu(), 1.0)

    # no backprop gradients
    def test(self):
        self.real_A_DIV = Variable(self.input_A_DIV)#, requires_grad=False)
        self.real_A_Vx = Variable(self.input_A_Vx)#, requires_grad=False)
        self.real_A_Vy = Variable(self.input_A_Vy)#, requires_grad=False)
        self.real_A_discrete = Variable(self.input_A, volatile=True)
        
        mask_var = Variable(self.mask.float(), volatile=True)
        self.fake_B_discrete = self.netG(torch.cat((self.real_A_discrete, mask_var), dim=1))
        self.fake_B_DIV = self.netG_DIV(self.fake_B_discrete)
        self.fake_B_Vx = self.netG_Vx(self.fake_B_discrete)
        self.fake_B_Vy = self.netG_Vy(self.fake_B_discrete)
        
        self.fake_B_classes = torch.max(self.fake_B_discrete, dim=1, keepdim=True)[1]

        self.real_B_DIV = Variable(self.input_B_DIV)#, requires_grad=False)
        self.real_B_Vx = Variable(self.input_B_Vx)#, requires_grad=False)
        self.real_B_Vy = Variable(self.input_B_Vy)#, requires_grad=False)
        self.real_B_discrete = Variable(self.input_B, volatile=True)

        self.real_B_classes = F.log_softmax(self.real_B_discrete, dim=1)
        self.real_B_classes = torch.max(self.real_B_discrete, dim=1, keepdim=True)[1]

        self.fake_B_one_hot = torch.zeros(self.fake_B_discrete.shape)

        self.fake_B_one_hot.scatter_(1, self.fake_B_classes.data.cpu(), 1.0)

    # get image paths
    def get_image_paths(self):
        return self.image_paths


    def calc_gradient_penalty(self, netD, real_data, fake_data):
        # print "real_data: ", real_data.size(), fake_data.size()
        alpha = torch.rand(real_data.shape[0], 1)
        alpha = alpha.expand(alpha.shape[0], real_data[0, ...].nelement()).contiguous().view(-1, *real_data.shape[1:])
        alpha = alpha.cuda(self.gpu_ids[0]) if len(self.gpu_ids) > 0 else alpha

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        if len(self.gpu_ids) > 0:
            interpolates = interpolates.cuda(self.gpu_ids[0])
        interpolates = autograd.Variable(interpolates, requires_grad=True)

        disc_interpolates = netD(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(self.gpu_ids[0]) if len(self.gpu_ids) > 0 else torch.ones(
                                      disc_interpolates.size()),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

        return gradient_penalty


    def backward_single_D(self, net_D, cond_data, real_data, fake_data):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # In this case real_A, the input, is our conditional vector
        fake_AB = torch.cat((cond_data, fake_data), dim=1)
        fake_loss = net_D(fake_AB.detach()).mean()
        # self.loss_D2_fake = self.criterionGAN(pred_fake, False)

        # Real
        real_AB = torch.cat((cond_data, real_data), dim=1)
        real_loss = net_D(real_AB).mean()
        # self.loss_D2_real = self.criterionGAN(pred_real, True)

        grad_pen = self.calc_gradient_penalty(net_D, real_AB.data, fake_AB.data)

        # Combined loss
        # self.loss_D2 = (self.loss_D2_fake + self.loss_D2_real) * 0.5
        loss = fake_loss - real_loss + grad_pen * self.opt.lambda_C

        loss.backward()

        return loss, real_loss, fake_loss


    def backward_D(self, net_Ds, optimisers, cond_data, real_data, fake_data):

        losses = torch.FloatTensor((len(net_Ds)))
        real_losses = torch.FloatTensor((len(net_Ds)))
        fake_losses = torch.FloatTensor((len(net_Ds)))

        for i, (net_D, optimiser) in enumerate(zip(net_Ds, optimisers)):
            optimiser.zero_grad()
            loss, real_loss, fake_loss = self.backward_single_D(
                net_D, cond_data, real_data, fake_data)
            loss.backward()
            optimiser.step()

            losses[i] = loss.data[0]
            real_losses[i] = real_loss.data[0]
            fake_losses[i] = fake_loss.data[0]

        return losses.mean(), real_losses.mean(), fake_losses.mean()


    def backward_G(self):
        # First, G(A) should fake the discriminator
        # Note that we don't detach here because we DO want to backpropagate
        # to the generator this time

        # fake_AB = torch.cat((self.real_A_discrete, self.fake_B_discrete), 1)
        fake_AB = torch.cat((self.real_A_discrete, self.fake_B_discrete), dim=1)
        pred_fake1 = torch.cat([netD1(fake_AB).mean() for netD1 in self.netD1s]).mean()
        
        # fake_AB = torch.cat((self.real_A_discrete, self.fake_B_DIV), 1)
        fake_AB = torch.cat((self.real_A_discrete, self.fake_B_DIV, self.fake_B_Vx, self.fake_B_Vy), dim=1)
        pred_fake2 = torch.cat([netD2(fake_AB).mean() for netD2 in self.netD2s]).mean()

        # We only optimise with respect to the fake prediction because
        # the first term (i.e. the real one) is independent of the generator i.e. it is just a constant term
        # self.loss_G_GAN1 = self.criterionGAN(pred_fake1, True)
        # self.loss_G_GAN2 = self.criterionGAN(pred_fake2, True)

        self.loss_G_GAN1 = -pred_fake1
        self.loss_G_GAN2 = -pred_fake2

        self.loss_G_GAN = self.loss_G_GAN1 + self.loss_G_GAN2

        self.loss_G_L2_DIV = self.criterionL2(
            self.fake_B_DIV.masked_select(self.mask).view(self.batch_size, 1, self.mask_size_y[0], self.mask_size_x[0]),
            self.real_B_DIV.masked_select(self.mask).view(self.batch_size, 1, self.mask_size_y[0], self.mask_size_x[0])) * self.opt.lambda_A
        self.loss_G_L2_Vx = self.criterionL2(
            self.fake_B_Vx.masked_select(self.mask).view(self.batch_size, 1, self.mask_size_y[0], self.mask_size_x[0]), 
            self.real_B_Vx.masked_select(self.mask).view(self.batch_size, 1, self.mask_size_y[0], self.mask_size_x[0])) * self.opt.lambda_A
        self.loss_G_L2_Vy = self.criterionL2(
            self.fake_B_Vy.masked_select(self.mask).view(self.batch_size, 1, self.mask_size_y[0], self.mask_size_x[0]),
            self.real_B_Vy.masked_select(self.mask).view(self.batch_size, 1, self.mask_size_y[0], self.mask_size_x[0])) * self.opt.lambda_A

        self.loss_G_L2 = self.loss_G_L2_DIV + self.loss_G_L2_Vx + self.loss_G_L2_Vy

        ce_fun = self.criterionCE()

        fake_B_discrete_masked = self.fake_B_discrete.masked_select(self.mask.repeat(1, 3, 1, 1)).view(
                self.batch_size, 3, self.mask_size_y[0], self.mask_size_x[0])
        real_B_classes_masked = self.real_B_classes.masked_select(self.mask.squeeze()).view(
                self.batch_size, self.mask_size_y[0], self.mask_size_x[0])

        # print(fake_B_discrete_masked)
        # print(real_B_classes_masked)

        self.loss_G_CE = ce_fun(F.log_softmax(fake_B_discrete_masked, dim=1),
            real_B_classes_masked) * self.opt.lambda_B

        self.loss_G = self.loss_G_GAN + self.loss_G_L2 + self.loss_G_CE

        self.loss_G.backward()


    def optimize_parameters(self, **kwargs):
        # Doesn't do anything with discriminator, just populates input (conditional), 
        # target and generated data in object
        self.forward()

        for _ in range(5 if kwargs['step_no'] >= 25 else 25):
            self.loss_D1, self.loss_D1_real, self.loss_D1_fake = self.backward_D(self.netD1s, self.optimizer_D1s, self.real_A_discrete,
                self.real_B_discrete, self.fake_B_discrete)

            self.loss_D2, self.loss_D2_real, self.loss_D2_fake = self.backward_D(self.netD2s, self.optimizer_D2s, self.real_A_discrete, 
                torch.cat((self.real_B_DIV, self.real_B_Vx, self.real_B_Vy), dim=1), 
                torch.cat((self.fake_B_DIV, self.fake_B_Vx, self.fake_B_Vy), dim=1))



        self.optimizer_G.zero_grad()
        self.optimizer_G_DIV.zero_grad()
        self.backward_G()
        self.optimizer_G.step()
        self.optimizer_G_DIV.step()
        self.optimizer_G_Vx.step()
        self.optimizer_G_Vy.step()


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

        mask_edge = roberts(self.mask.data.cpu().numpy().squeeze()[0, ...])
        mask_edge_coords = np.where(mask_edge)

        real_A_discrete = util.tensor2im(self.real_A_discrete.data)
        real_A_discrete[mask_edge_coords] = np.max(real_A_discrete)

        real_A_DIV = util.tensor2im(self.real_A_DIV.data)
        real_A_DIV[mask_edge_coords] = np.max(real_A_DIV)

        real_A_Vx = util.tensor2im(self.real_A_Vx.data)
        real_A_Vx[mask_edge_coords] = np.max(real_A_Vx)

        real_A_Vy = util.tensor2im(self.real_A_Vy.data)
        real_A_Vy[mask_edge_coords] = np.max(real_A_Vy)

        real_B_discrete = util.tensor2im(self.real_B_discrete.data)
        real_B_discrete[mask_edge_coords] = np.max(real_B_discrete)

        real_B_DIV = util.tensor2im(self.real_B_DIV.data)
        real_B_DIV[mask_edge_coords] = np.max(real_B_DIV)

        real_B_Vx = util.tensor2im(self.real_B_Vx.data)
        real_B_Vx[mask_edge_coords] = np.max(real_B_Vx)

        real_B_Vy = util.tensor2im(self.real_B_Vy.data)
        real_B_Vy[mask_edge_coords] = np.max(real_B_Vy)

        fake_B_discrete = util.tensor2im(F.log_softmax(self.fake_B_discrete, dim=1).data)
        fake_B_discrete[mask_edge_coords] = np.max(fake_B_discrete)

        fake_B_one_hot = util.tensor2im(self.fake_B_one_hot)
        fake_B_one_hot[mask_edge_coords] = np.max(fake_B_one_hot)

        fake_B_DIV = util.tensor2im(self.fake_B_DIV.data)
        fake_B_DIV[mask_edge_coords] = np.max(fake_B_DIV)

        fake_B_Vx = util.tensor2im(self.fake_B_Vx.data)
        fake_B_Vx[mask_edge_coords] = np.max(fake_B_Vx)

        fake_B_Vy = util.tensor2im(self.fake_B_Vy.data)
        fake_B_Vy[mask_edge_coords] = np.max(fake_B_Vy)

        return OrderedDict([
            ('input_one_hot', real_A_discrete),
            ('output_softmax', fake_B_discrete),
            ('output_one_hot', fake_B_one_hot),
            ('ground_truth_one_hot', real_B_discrete),
            ('input_divergence', real_A_DIV), 
            ('output_divergence', fake_B_DIV),
            ('ground_truth_divergence', real_B_DIV), 
            ('input_Vx', real_A_Vx), 
            ('output_Vx', fake_B_Vx),
            ('ground_truth_Vx', real_B_Vx), 
            ('input_Vy', real_A_Vy), 
            ('output_Vy', fake_B_Vy),
            ('ground_truth_Vy', real_B_Vy), 
            ])


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

        mask_coords = np.where(self.mask.numpy().squeeze())
        mask_tl =(mask_coords[0][0], mask_coords[1][0])
        mask_br =(mask_coords[0][-1], mask_coords[1][-1])

        d_h_recall = 0.0
        d_h_precision = 0.0
        d_h_s = 0.0

        for c in np.unique(self.real_B_classes.data.numpy()):
            fake_channel = self.fake_B_one_hot.numpy().squeeze()[c]
            real_channel = self.real_B_discrete.data.numpy().squeeze()[c]
            # num_true_pos = np.sum(np.logical_and(fake_channel, fake_channel == real_channel).ravel())
            # num_true_neg = np.sum(np.logical_and(1-fake_channel, fake_channel == real_channel).ravel())
            # num_false_pos = np.sum((fake_channel > real_channel).ravel())
            # num_false_neg = np.sum((fake_channel < real_channel).ravel())

            # plt.subplot(241)
            # io.imshow(self.mask.numpy().squeeze())
            # plt.subplot(242)
            # io.imshow(fake_channel)
            # plt.subplot(243)
            # io.imshow(real_channel)
            # plt.subplot(244)
            # io.imshow((fake_channel == real_channel))
            # plt.subplot(245)
            # io.imshow(np.logical_and(fake_channel, fake_channel == real_channel))
            # plt.subplot(246)
            # io.imshow(np.logical_and(1 - fake_channel, fake_channel == real_channel))
            # plt.subplot(247)
            # io.imshow(fake_channel > real_channel)
            # plt.subplot(248)
            # io.imshow(fake_channel < real_channel)
            # plt.show()

            # print(num_true_pos)
            # print(num_true_neg)
            # print(num_false_pos)
            # print(num_false_neg)

            # num_true_pos, num_true_neg, num_false_pos, num_false_neg = get_stats(real_channel, fake_channel)

            # precision = num_true_pos / (num_true_pos + num_false_pos + eps)
            # metrics.append(('Class {} Precision'.format(c),  precision))
            # precisions.append(precision)
            
            # recall = num_true_pos / (num_true_pos + num_false_neg + eps)
            # metrics.append(('Class {} Recall'.format(c),  recall))
            # recalls.append(recall)

            fake_channel = fake_channel[mask_tl[0]:mask_br[0], mask_tl[1]:mask_br[1]]
            real_channel = real_channel[mask_tl[0]:mask_br[0], mask_tl[1]:mask_br[1]]

            # plt.subplot(121)
            # io.imshow(fake_channel)
            # plt.subplot(122)
            # io.imshow(real_channel)
            # plt.show()

            # num_true_pos, num_true_neg, num_false_pos, num_false_neg = get_stats(real_channel, fake_channel)

            # precision = (num_true_pos) / (num_true_pos + num_false_pos + eps)
            # metrics.append(('Class {} Precision (Inpainted region)'.format(c),  precision))
            # inpaint_precisions.append(precision)
            
            # recall = (num_true_pos) / (num_true_pos + num_false_neg + eps)
            # metrics.append(('Class {} Recall (Inpainted region)'.format(c),  recall))
            # inpaint_recalls.append(recall)



            # u : (M,N) ndarray
            #     Input array.
            # v : (O,N) ndarray
            #     Input array.
            # seed : int or None
            #     Local np.random.RandomState seed. Default is 0, a random shuffling of u and v that guarantees reproducibility.
            # Returns:    
            # d : double
            #     The directed Hausdorff distance between arrays u and v,
            # index_1 : int
            #     index of point contributing to Hausdorff pair in u
            # index_2 : int
            #     index of point contributing to Hausdorff pair in v
            # fake_coords = np.array(list(zip(*np.where(fake_channel))))
            # real_coords = np.array(list(zip(*np.where(real_channel))))
            fake_coords = np.array(np.where(fake_channel)).T
            real_coords = np.array(np.where(real_channel)).T

            # print(fake_coords)
            # print(real_coords)

            if not fake_coords.any() or not real_coords.any():
                mask_diagonal = euclidean(mask_br, mask_tl)
                d_h_fr = mask_diagonal
                d_h_rf = mask_diagonal
                d_h_s = mask_diagonal
            elif fake_coords.any() and real_coords.any():
                d_h_fr, i1_fr, i2_fr = directed_hausdorff(fake_coords, real_coords)
                d_h_rf, i1_rf, i2_rf = directed_hausdorff(real_coords, fake_coords)

                # f_y, f_x = fake_coords[i1_fr][0], fake_coords[i1_fr][1]
                # r_y, r_x = real_coords[i2_fr][0], real_coords[i2_fr][1]

                # pixel_layer = np.zeros(fake_channel.shape)
                # pixel_layer[f_y, f_x] = 1
                # pixel_layer[r_y, r_x] = 1
                # overlay_dh_fr = np.stack((fake_channel, real_channel, pixel_layer), axis=2)

                # f_y, f_x = fake_coords[i2_rf][0], fake_coords[i2_rf][1]
                # r_y, r_x = real_coords[i1_rf][0], real_coords[i1_rf][1]

                # pixel_layer = np.zeros(fake_channel.shape)
                # pixel_layer[f_y, f_x] = 1
                # pixel_layer[r_y, r_x] = 1
                # overlay_dh_rf = np.stack((fake_channel, real_channel, pixel_layer), axis=2)

                # plt.subplot(211)
                # io.imshow(overlay_dh_rf)
                # plt.title('Real to fake (recall)')
                # plt.subplot(212)
                # io.imshow(overlay_dh_fr)
                # plt.title('Fake to real (precision)')
                # plt.suptitle('Real - green, fake - red, fake pix - m, real pix - c\n'+
                #     'Recall {}, precision {}'.format(d_h_rf, d_h_fr))
                # plt.show()

                d_h_s = max(d_h_s, max(d_h_fr, d_h_rf))

            d_h_recall = max(d_h_recall, d_h_rf)
            d_h_precision = max(d_h_precision, d_h_fr)
            # print(d_h_fr)
            # print(d_h_rf)
            # i1, i2 = (i1_fr, i2_fr) if d_h_fr > d_h_rf else (i2_rf, i1_rf)

            # print(i1)
            # print(i2)
            # print(fake_coords[i1])
            # print(real_coords[i2])


        metrics.append(('Hausdorff distance (R)', d_h_recall))
        metrics.append(('Hausdorff distance (P)', d_h_precision))
        metrics.append(('Hausdorff distance (S)', d_h_s))

        # metrics.append(('Average precision', np.mean(precisions)))
        # metrics.append(('Average recall', np.mean(recalls)))

        # metrics.append(('Average precision (inpainted region)', np.mean(inpaint_precisions)))
        # metrics.append(('Average recall (inpainted region)', np.mean(inpaint_recalls)))

        return OrderedDict(metrics)


    def accumulate_metrics(self, metrics):
        d_h_recall = []
        d_h_precision = []
        d_h_s = []

        for metric in metrics:
            d_h_recall.append(metric['Hausdorff distance (R)'])
            d_h_precision.append(metric['Hausdorff distance (P)'])
            d_h_s.append(metric['Hausdorff distance (S)'])


        return OrderedDict([
            ('Hausdorff distance (R)', np.mean(d_h_recall)),
            ('Hausdorff distance (P)', np.mean(d_h_precision)),
            ('Hausdorff distance (S)', np.mean(d_h_s)),
            ])


    def save(self, label):
        self.save_network(self.netG, 'G', label, self.gpu_ids)
        self.save_network(self.netG_DIV, 'G_DIV', label, self.gpu_ids)
        self.save_network(self.netG_Vx, 'G_Vx', label, self.gpu_ids)
        self.save_network(self.netG_Vy, 'G_Vy', label, self.gpu_ids)

        for i in range(len(self.netD1s))
            self.save_network(self.netD1s[i], 'D1_%d' % i, label, self.gpu_ids)
            self.save_network(self.netD2s[i], 'D2_%d' % i, label, self.gpu_ids)

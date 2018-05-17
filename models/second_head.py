import torch
import torch.nn as nn
from torch.autograd import Variable


class SecondHead(nn.Module):
	def __init__(self, net, out_channels, module_type='ConvTranspose', depth=3, down_module_type=None):
		super(SecondHead, self).__init__()

		def get_output_hook(self, input, output):
			self.div_start_input = input
			self.div_start_output = output


		def walk_back_upconv(net, depth):
			# Walks back into network, until it finds the nth module of designated type
			# and returns preceding layer
			num_modules = 0
			for idx, module in enumerate(reversed(list(net.modules()))):
				if module_type in str(type(module)):
					num_modules += 1
				
				if num_modules == depth:
					return idx+1

		def store_conv_output(self, input, output):
			self.downconv_output = output

		self.depth = depth
		self.out_channels = out_channels
		self.no_skip_layer = down_module_type == None

		if not self.no_skip_layer:
			self.down_modules = [module for module in net.modules() if down_module_type in str(type(module))]

			for i, module in enumerate(self.down_modules):
				module.register_forward_hook(store_conv_output)


		last_layer_idx = walk_back_upconv(net, self.depth)
		last_layer = list(net.modules())[-last_layer_idx]

		self.inner_layer = list(net.modules())[-last_layer_idx-1]

		self.inner_layer.register_forward_hook(get_output_hook)

		self.second_head = None

		if self.no_skip_layer:
			modules = []

			in_channels = last_layer.in_channels

			for i in range(self.depth-1):
				modules.append(nn.ConvTranspose2d(in_channels, in_channels / 2, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False))
				modules.append(nn.BatchNorm2d(in_channels / 2, eps=1e-05, momentum=0.1, affine=True))
				modules.append(nn.ReLU(inplace=True))

				in_channels /= 2

			modules.append(nn.ConvTranspose2d(in_channels, self.out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)))
			modules.append(nn.Tanh())

			self.model = nn.Sequential(*modules)
		else:
			modules = []

			in_channels = last_layer.in_channels

			for i in range(self.depth-1):
				modules.append(nn.Sequential(
					nn.ConvTranspose2d(in_channels, int(in_channels / 4), kernel_size=(4, 4), stride=(2, 2), padding=(1, 1), bias=False),
					nn.BatchNorm2d(int(in_channels / 4), eps=1e-05, momentum=0.1, affine=True),
					nn.ReLU(inplace=True)))

				in_channels = int(in_channels / 2)

			modules.append(nn.Sequential(
				nn.ConvTranspose2d(in_channels, self.out_channels, kernel_size=(4, 4), stride=(2, 2), padding=(1, 1)),
				nn.Tanh()))

			self.model = nn.Sequential(*modules)

	def forward(self, x):
		if self.no_skip_layer:
			return self.model(self.inner_layer.div_start_output)
		else:
			y = self.inner_layer.div_start_output

			for i, module in enumerate(self.model):
				if i == 0:
					y = module(y)
				else:
					y = torch.cat((y, self.down_modules[self.depth-i-1].downconv_output), dim=1)

					y = module(y)

			return y


if __name__ == '__main__':
	import models.networks as networks

	G = networks.define_G(3, 3, 32, 'unet_256', 'batch', True, 'normal', [])


	x = torch.randn((1, 3, 256, 512))
	x_var = Variable(x)

	s1 = SecondHead(G, 'ConvTranspose2d', 3, 'Conv2d')
	s2 = SecondHead(G, 'ConvTranspose2d', 3)

	y = G(x_var)

	print(s1(y).shape)

	print(s1)

	print(s2(y).shape)
import torch.nn as nn


class ConnCompNet(nn.Module):
	def __init__(self, im_size, hist_size, nf=64, num_layers=6):
		super(ConnCompNet, self).__init__()
		layers = []

		in_channels = im_size[2]
		for i in range(num_layers):
			num_filters = nf*(num_layers - i)

			layers.append(nn.Conv2d(in_channels, num_filters, kernel_size=4, stride=2, padding=(1, 1)))
			layers.append(nn.LeakyReLU(0.2))
			layers.append(nn.BatchNorm2d(num_filters))

			in_channels = num_filters


		fc_size = int((im_size[0] * im_size[1]) / 2**num_layers)
		self.fc = nn.Linear(fc_size, hist_size)

		self.model = nn.Sequential(*layers)


	def forward(self, x, **kwargs):
		y = self.model(x)

		y = self.fc(y.view(x.shape[0], -1))

		return y


if __name__ == '__main__':
	import torch
	
	c = ConnCompNet((256, 512, 3), 128, 64)

	x = torch.autograd.Variable(torch.randn(1, 3, 256, 512))

	y = c(x)

	print(y.shape)
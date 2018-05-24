import torch

class SparseBatchNorm2d(torch.nn.BatchNorm2d):
	def forward(self, x):
		output = super(SparseBatchNorm2d, self).forward(x)
		output.sparse_mask = x.sparse_mask

		return output

class SparseInstanceNorm2d(torch.nn.InstanceNorm2d):
	def forward(self, x):
		output = super(SparseInstanceNorm2d, self).forward(x)
		output.sparse_mask = x.sparse_mask

		return output

class SparseLeakyReLU(torch.nn.LeakyReLU):
	def forward(self, x):
		output = super(SparseLeakyReLU, self).forward(x)
		output.sparse_mask = x.sparse_mask

		return output

class SparseReLU(torch.nn.ReLU):
	def forward(self, x):
		output = super(SparseReLU, self).forward(x)
		output.sparse_mask = x.sparse_mask

		return output

class SparseTanh(torch.nn.Tanh):
	def forward(self, x):
		output = super(SparseTanh, self).forward(x)
		output.sparse_mask = x.sparse_mask

		return output

class SparseConvTranspose2d(torch.nn.ConvTranspose2d):
	## torch.nn.Upsample
	## followed by SparseConv2d
	def forward(self, x):
		output = super(SparseConvTranspose2d, self).forward(x)
		output.sparse_mask = x.sparse_mask

		return output


class SparseConv2d(torch.nn.Conv2d):
	def __init__(self, *args, **kwargs):
		if 'bias' in kwargs.keys():
			assert(kwargs['bias'] == False)
		else:
			kwargs['bias'] = False

		super().__init__(*args, **kwargs)

		self.mask_weight_conv = torch.nn.Conv2d(self.in_channels, 1, self.kernel_size,
			self.stride, self.padding, self.dilation, bias=False)
		self.mask_weight_conv.weight.data.fill_(1)

		self.maxpool = torch.nn.MaxPool2d(self.kernel_size, self.stride,
			self.padding, self.dilation)


	def forward(self, input):
		if 'sparse_mask' not in dir(input):
			return super(SparseConv2d, self).forward(input)

		mask = input.sparse_mask
		output = super(SparseConv2d, self).forward(input * mask.float())
		mask_weights = self.mask_weight_conv(mask.float())

		output = output / mask_weights
		output.sparse_mask = torch.min(self.maxpool(mask), dim=1, keepdim=True)[0]

		return output


import io
import pytest
import torch

try:
	import mock
except ImportError:
	import unittest.mock as mock

from data.geo_dataset import GeoDataset, get_dat_files
import models
from models.pix2pix_geo_model import Pix2PixGeoModel

from options.base_options import BaseOptions

from mock_utils import *

class NullOptions(object):
	pass


def test_generator():
	MockGenerator = mock.MagicMock()

	def mock_define_G(*args, **kwargs):
		return MockGenerator


	torch_mock = mock.patch('torch.optim')
	cat_mock = mock.patch('torch.cat', new=cat_stub)
	softmax_mock = mock.patch('torch.nn.functional.log_softmax', new=softmax_stub)
	max_mock = mock.patch('torch.max', new=max_stub)
	zeros_mock = mock.patch('torch.zeros', new=zeros_stub)
	scheduler_mock = mock.patch('models.networks.get_scheduler')

	torch_mock.start()
	cat_mock.start()
	scheduler_mock.start()
	softmax_mock.start()
	max_mock.start()
	zeros_mock.start()


	with mock.patch.object(torch.autograd, 'Variable', new=variable_stub):
		with mock.patch('models.networks.define_G', return_value=MockGenerator):
			gan = Pix2PixGeoModel()
			opt = NullOptions()
			opt.gpu_ids = []
			opt.isTrain = True
			opt.checkpoints_dir = ''
			opt.name = 'test_run'
			opt.input_nc = 3
			opt.output_nc = 3
			opt.ngf = 420
			opt.ndf = 69
			opt.which_model_netG = 'the perfect working model'
			opt.norm = 'batch'
			opt.no_dropout = False
			opt.init_type = 'rando'
			opt.discrete_only = True
			opt.num_discrims = 0
			opt.no_mask_to_critic = False
			opt.which_model_netD = 'wgan-gp'
			opt.no_lsgan = True
			opt.n_layers_D = 3
			opt.continue_train = False
			opt.lr = 0.01
			opt.beta1 = 0.9
			opt.which_direction = 'AtoB'

			gan.initialize(opt)

			fake_dataset = DatasetMock()
			# fake_dataset = mock.MagicMock()

			def stringify(self, arg):
				return StringMock()

			# torch.autograd.Variable = identity

			# fake_dataset.__getitem__ = stringify
			gan.set_input(fake_dataset)
			gan.forward()

			# print(MockGenerator.call_args)
			assert(MockGenerator.call_args[0][0] == (['A', 'mask_float']))
			# MockGenerator.assert_called_with()
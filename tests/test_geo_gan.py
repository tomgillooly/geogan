import io
import pytest
import torch

from data.geo_dataset import GeoDataset, get_dat_files
import models
from models.pix2pix_geo_model import Pix2PixGeoModel

from options.base_options import BaseOptions

from mock_utils import *

class NullOptions(object):
	pass

def standard_options():
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
	opt.which_model_netD = 'good d'
	opt.no_lsgan = True
	opt.n_layers_D = 3
	opt.continue_train = False
	opt.lr = 0.01
	opt.beta1 = 0.9
	opt.which_direction = 'AtoB'

	return opt


@pytest.fixture
def basic_gan(mocker):
	v = VariableMock()
	v.name = 'fake_discrete_output'
	MockGenerator = mocker.MagicMock(return_value=v)

	v = VariableMock()
	v.name = 'discrim_output'
	MockDiscriminator = mocker.MagicMock(return_value=v)


	mocker.patch('torch.optim')
	mocker.patch('models.networks.get_scheduler')
	mocker.patch('torch.cat', new=cat_stub)
	mocker.patch('torch.nn.functional.log_softmax', new=softmax_stub)
	mocker.patch('torch.max', new=max_stub)
	mocker.patch('torch.zeros', new=zeros_stub)
	mocker.patch.object(torch.autograd, 'Variable', new=variable_stub)
	mocker.patch('models.networks.define_G', return_value=MockGenerator)
	mocker.patch('models.networks.define_D', new=MockDiscriminator)

	return MockGenerator


def test_generator(basic_gan):
	MockGenerator = basic_gan

	gan = Pix2PixGeoModel()

	opt = standard_options()
	gan.initialize(opt)

	fake_dataset = DatasetMock()

	gan.set_input(fake_dataset)
	gan.forward()

	assert(MockGenerator.call_args[0][0] == (['A', 'mask_float']))
	print(MockGenerator.call_args)


def test_generator_discriminator(basic_gan):
	MockGenerator = basic_gan

	gan = Pix2PixGeoModel()

	opt = standard_options()
	opt.num_discrims = 1


	gan.initialize(opt)

	fake_dataset = DatasetMock()

	gan.set_input(fake_dataset)
	gan.forward()

	assert(MockGenerator.call_args[0][0] == (['A', 'mask_float']))
	

	for netD1 in gan.netD1s:
		assert(not netD1.called)

	for netD2 in gan.netD1s:
		assert(not netD1.called)
		# assert(netD1.call_args[0][0] == )




def test_generator_w_continuous(basic_gan):
	MockGenerator = basic_gan

	gan = Pix2PixGeoModel()

	opt = standard_options()
	opt.discrete_only = False

	gan.initialize(opt)

	fake_dataset = DatasetMock()

	gan.netG_DIV = basic_gan.mocker.MagicMock()
	gan.netG_Vx = basic_gan.mocker.MagicMock()
	gan.netG_Vy = basic_gan.mocker.MagicMock()

	gan.set_input(fake_dataset)
	gan.forward()

	# call args structure is ((list of args as tuple), {list of keyword args})
	# So [0][0] accesses first non-keyword arg
	assert(gan.netG.call_args[0][0] == ['A', 'mask_float'])
	assert(gan.netG_DIV.call_args[0][0].name == 'fake_discrete_output')
	assert(gan.netG_Vx.call_args[0][0].name == 'fake_discrete_output')
	assert(gan.netG_Vy.call_args[0][0].name == 'fake_discrete_output')
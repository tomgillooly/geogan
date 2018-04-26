import io
import pytest
import torch

from data.geo_dataset import GeoDataset, get_dat_files
import models
from models.pix2pix_geo_model import Pix2PixGeoModel, get_innermost

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
	opt.continent_data = False

	return opt


def fake_network(mocker, name):
	v = VariableMock()
	v.name = name
	return mocker.MagicMock(return_value=v)



@pytest.fixture
def basic_gan(mocker):
	MockGenerator = fake_network(mocker, 'fake_discrete_output')
	
	def fake_discrim(*args, **kwargs):
		return fake_network(mocker, 'discrim_output')

	mocker.patch('torch.optim')
	mocker.patch('models.networks.get_scheduler')
	mocker.patch('torch.cat', new=cat_stub)
	mocker.patch('torch.nn.NLLLoss2d', new=nll_loss2d_stub)
	mocker.patch('torch.nn.MSELoss', new=mse_loss_stub)
	mocker.patch('torch.nn.functional.log_softmax', new=softmax_stub)
	mocker.patch('torch.nn.functional.binary_cross_entropy', new=bce_loss_stub)
	mocker.patch('torch.max', new=max_stub)
	mocker.patch('torch.zeros', new=zeros_stub)
	mocker.patch('torch.ones', new=ones_stub)
	mocker.patch('torch.mean', new=mean_stub)
	mocker.patch.object(torch.autograd, 'Variable', new=variable_stub)
	mocker.patch('models.networks.define_G', return_value=MockGenerator)

	# define_D_config = {'__call__.side_effect': fake_discrim}
	mocker.patch('models.networks.define_D', side_effect=fake_discrim)

	return MockGenerator


def test_generator(basic_gan):
	MockGenerator = basic_gan

	gan = Pix2PixGeoModel()

	opt = standard_options()
	gan.initialize(opt)

	fake_dataset = DatasetMock()

	gan.set_input(fake_dataset)
	gan.forward()

	print(MockGenerator.call_args)
	assert(MockGenerator.call_args[0][0].name == (['A', 'mask_float']))


def test_generator_discriminator(basic_gan, mocker):
	MockGenerator = basic_gan

	gan = Pix2PixGeoModel()

	opt = standard_options()
	opt.num_discrims = 1
	opt.low_iter = 1
	opt.high_iter = 1

	opt.lambda_A = 1
	opt.lambda_B = 1
	opt.lambda_C = 1
	opt.discrete_only = False
	opt.local_loss = False
	opt.weighted_ce = False

	gan.initialize(opt)

	assert(all([netD1 != netD2 for netD1, netD2 in zip(gan.netD1s, gan.netD2s)]))

	gan.netG_DIV = fake_network(mocker, 'fake_DIV')
	gan.netG_Vx = fake_network(mocker, 'fake_Vx')
	gan.netG_Vy = fake_network(mocker, 'fake_Vy')

	fake_dataset = DatasetMock()

	gan.set_input(fake_dataset)

	# gan.backward_D = mocker.MagicMock(return_value=('loss', 'loss', 'loss', 'loss'))
	# gan.backward_single_D = mocker.MagicMock(return_value=VariableMock().set_name('loss'))

	gan.optimize_parameters(step_no=1)

	assert(MockGenerator.call_args[0][0].name == (['A', 'mask_float']))
	assert(gan.netG_DIV.call_args[0][0].name == ('fake_discrete_output'))
	assert(gan.netG_Vx.call_args[0][0].name == ('fake_discrete_output'))
	assert(gan.netG_Vy.call_args[0][0].name == ('fake_discrete_output'))

	# First call -> non-keyword args -> argument index
	# assert(gan.backward_D.call_args_list[0][0][0] == gan.netD1s)
	# assert(gan.backward_D.call_args_list[0][0][2].name == ['A', 'mask_float'])
	# assert(gan.backward_D.call_args_list[0][0][3].name == 'B')
	# assert(gan.backward_D.call_args_list[0][0][4].name == 'fake_discrete_output')

	# assert(gan.backward_D.call_args_list[1][0][0] == gan.netD2s)
	# assert(gan.backward_D.call_args_list[1][0][2].name == ['A', 'mask_float'])
	# assert(gan.backward_D.call_args_list[1][0][3].name == ['B_DIV', 'B_Vx', 'B_Vy'])
	# assert(gan.backward_D.call_args_list[1][0][4].name == ['fake_DIV', 'fake_Vx', 'fake_Vy'])

	# assert(gan.backward_single_D.call_args_list[0][0][0] == gan.netD1s[0])
	# assert(gan.backward_single_D.call_args_list[0][0][1].name == ['A', 'mask_float'])
	# assert(gan.backward_single_D.call_args_list[0][0][2].name == 'B')
	# assert(gan.backward_single_D.call_args_list[0][0][3].name == 'fake_discrete_output')

	# assert(gan.backward_single_D.call_args_list[1][0][0] == gan.netD2s[0])
	# assert(gan.backward_single_D.call_args_list[1][0][1].name == ['A', 'mask_float'])
	# assert(gan.backward_single_D.call_args_list[1][0][2].name == ['B_DIV', 'B_Vx', 'B_Vy'])
	# assert(gan.backward_single_D.call_args_list[1][0][3].name == ['fake_DIV', 'fake_Vx', 'fake_Vy'])


	for netD1 in gan.netD1s:
		assert(netD1.call_args_list[0][0][0].name == '[A, mask_float, fake_discrete_output]_detach')
		assert(netD1.call_args_list[1][0][0].name == ['A', 'mask_float', 'B'])
	
		# Third call to grad penalty??

	for netD2 in gan.netD2s:
		assert(netD2.call_args_list[0][0][0].name == '[A, mask_float, fake_DIV, fake_Vx, fake_Vy]_detach')
		assert(netD2.call_args_list[1][0][0].name == ['A', 'mask_float', 'B_DIV', 'B_Vx', 'B_Vy'])

		# Third call to grad penalty??


def test_with_continents(basic_gan, mocker):
	MockGenerator = basic_gan

	gan = Pix2PixGeoModel()

	opt = standard_options()
	opt.num_discrims = 1
	opt.low_iter = 1
	opt.high_iter = 1

	opt.lambda_A = 1
	opt.lambda_B = 1
	opt.lambda_C = 1
	opt.discrete_only = False
	opt.local_loss = False
	opt.weighted_ce = False
	opt.continent_data = True

	gan.initialize(opt)

	assert(all([netD1 != netD2 for netD1, netD2 in zip(gan.netD1s, gan.netD2s)]))

	gan.netG_DIV = fake_network(mocker, 'fake_DIV')
	gan.netG_Vx = fake_network(mocker, 'fake_Vx')
	gan.netG_Vy = fake_network(mocker, 'fake_Vy')

	fake_dataset = DatasetMock()

	gan.set_input(fake_dataset)

	gan.optimize_parameters(step_no=1)

	assert(MockGenerator.call_args[0][0].name == (['A', 'mask_float', 'continents_float']))
	assert(gan.netG_DIV.call_args[0][0].name == ('fake_discrete_output'))
	assert(gan.netG_Vx.call_args[0][0].name == ('fake_discrete_output'))
	assert(gan.netG_Vy.call_args[0][0].name == ('fake_discrete_output'))


	for netD1 in gan.netD1s:
		assert(netD1.call_args_list[0][0][0].name == '[A, mask_float, continents_float, fake_discrete_output]_detach')
		assert(netD1.call_args_list[1][0][0].name == ['A', 'mask_float', 'continents_float', 'B'])
	
		# Third call to grad penalty??

	for netD2 in gan.netD2s:
		assert(netD2.call_args_list[0][0][0].name == '[A, mask_float, continents_float, fake_DIV, fake_Vx, fake_Vy]_detach')
		assert(netD2.call_args_list[1][0][0].name == ['A', 'mask_float', 'continents_float', 'B_DIV', 'B_Vx', 'B_Vy'])

		# Third call to grad penalty??



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
	assert(gan.netG.call_args[0][0].name == ['A', 'mask_float'])
	assert(gan.netG_DIV.call_args[0][0].name == 'fake_discrete_output')
	assert(gan.netG_Vx.call_args[0][0].name == 'fake_discrete_output')
	assert(gan.netG_Vy.call_args[0][0].name == 'fake_discrete_output')


def test_innermost_block_retrieval():
	m1 = torch.nn.Module()
	m2 = torch.nn.Module()
	m3 = torch.nn.Module()

	inner_model = torch.nn.Sequential(m1, m2, m3)

	outer_model = torch.nn.Sequential(torch.nn.Module(), inner_model, torch.nn.Module())
	outer_model = torch.nn.Sequential(outer_model)

	assert(get_innermost(outer_model) == m2)
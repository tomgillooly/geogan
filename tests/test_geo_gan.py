import io
import pytest
import torch

import models
from models.pix2pix_geo_model import Pix2PixGeoModel, get_innermost
from models.networks import UnetGenerator

from options.base_options import BaseOptions

from mock_utils import *

from functools import reduce

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
	opt.num_folders = 1
	opt.folder_pred=False
	
	opt.high_iter = 1
	opt.low_iter = 1
	opt.lambda_A = 1
	opt.lambda_B = 1
	opt.lambda_C = 1

	opt.local_loss = False
	opt.div_only = False

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
	mocker.patch('torch.sum', new=sum_stub)
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

	gan.initialize(opt)

	assert(models.networks.define_D.called)
	assert(models.networks.define_D.call_args[0][0] == 7)

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
	opt.continent_data = True

	gan.initialize(opt)

	assert(all([netD1 != netD2 for netD1, netD2 in zip(gan.netD1s, gan.netD2s)]))

	gan.netG_DIV = fake_network(mocker, 'fake_DIV')
	gan.netG_Vx = fake_network(mocker, 'fake_Vx')
	gan.netG_Vy = fake_network(mocker, 'fake_Vy')

	fake_dataset = DatasetMock()

	gan.set_input(fake_dataset)

	gan.optimize_parameters(step_no=1)

	assert(MockGenerator.call_args[0][0].name == (['A', 'mask_float', 'cont_float']))
	assert(gan.netG_DIV.call_args[0][0].name == ('fake_discrete_output'))
	assert(gan.netG_Vx.call_args[0][0].name == ('fake_discrete_output'))
	assert(gan.netG_Vy.call_args[0][0].name == ('fake_discrete_output'))


	for netD1 in gan.netD1s:
		assert(netD1.call_args_list[0][0][0].name == '[A, mask_float, cont_float, fake_discrete_output]_detach')
		assert(netD1.call_args_list[1][0][0].name == ['A', 'mask_float', 'cont_float', 'B'])
	
		# Third call to grad penalty??

	for netD2 in gan.netD2s:
		assert(netD2.call_args_list[0][0][0].name == '[A, mask_float, cont_float, fake_DIV, fake_Vx, fake_Vy]_detach')
		assert(netD2.call_args_list[1][0][0].name == ['A', 'mask_float', 'cont_float', 'B_DIV', 'B_Vx', 'B_Vy'])

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


def test_splitting_unet():
	unet = UnetGenerator(3, 3, 1)

	x = torch.randn((1, 3, 64, 64))

	def extract_latent_vector(self, input, output):
		self.latent_vector = output

	inner = get_innermost(unet, 'UnetSkipConnectionBlock')
	inner.register_forward_hook(extract_latent_vector)

	y = unet(x)
	
	fc1 = torch.nn.Linear(2*2*64*8, 20)
	loss_fun = torch.nn.CrossEntropyLoss()

	gt = torch.LongTensor((1,))
	gt[0] = 3

	print(inner.latent_vector.shape)
	
	folder_vec = fc1(inner.latent_vector.view(1, -1)).expand(1, -1)
	
	loss = loss_fun(folder_vec, gt)

	print(loss)


def test_get_downsample():
	unet = UnetGenerator(3, 3, 2)

	downsample = models.pix2pix_geo_model.get_downsample(unet)
	inner = get_innermost(unet, 'UnetSkipConnectionBlock')
	inner.register_forward_hook(models.pix2pix_geo_model.save_output_hook)

	x = torch.rand((1, 3, 256, 512))

	y = unet(x)

	inner_vector = inner.output
	
	assert(inner_vector.shape[2] == 256 / downsample)
	assert(inner_vector.shape[3] == 512 / downsample)
	
	unet = UnetGenerator(3, 3, 7, 64)

	downsample = models.pix2pix_geo_model.get_downsample(unet)
	inner = get_innermost(unet, 'UnetSkipConnectionBlock')
	inner.register_forward_hook(models.pix2pix_geo_model.save_output_hook)

	x = torch.rand((1, 3, 256, 512))

	y = unet(x)

	inner_vector = inner.output
	
	assert(inner_vector.shape[2] == 256 / downsample)
	assert(inner_vector.shape[3] == 512 / downsample)


def test_correct_shape_for_folder_id(mocker):
	# MockGenerator = basic_gan

	mocker.patch('torch.optim')
	mocker.patch('models.networks.get_scheduler')
	mocker.patch('models.networks.define_D', side_effect=mocker.MagicMock())
	mocker.patch('models.networks.define_G', return_value=UnetGenerator(3, 3, 7, 4))

	gan = Pix2PixGeoModel()

	opt = standard_options()
	opt.num_discrims = 1
	opt.low_iter = 1
	opt.high_iter = 1

	opt.which_model_netD = 'wgan-gp'
	opt.lambda_A = 1
	opt.lambda_B = 1
	opt.lambda_C = 1
	opt.discrete_only = False
	opt.local_loss = False
	opt.fineSize = 256
	opt.ngf = 4
	opt.num_folders = 20
	opt.folder_pred = True

	gan.initialize(opt)

	x = torch.rand((2, 3, 256, 512))

	y = gan.netG(x)

	folder = gan.folder_fc(gan.netG.inner_layer.output.view(2, -1))

	assert(folder.shape == (2, 20))

def test_folder_id_used_in_cross_entropy_loss(basic_gan, mocker):
	MockGenerator = basic_gan
	MockInnerLayer = fake_network(mocker, 'innermost')

	mocker.patch('models.pix2pix_geo_model.get_innermost',
		new=mocker.MagicMock(return_value=MockInnerLayer))

	gan = Pix2PixGeoModel()

	opt = standard_options()
	opt.num_discrims = 1
	opt.low_iter = 1
	opt.high_iter = 1

	opt.which_model_netD = 'cwgan-gp'
	opt.lambda_A = 1
	opt.lambda_B = 1
	opt.lambda_C = 1
	opt.discrete_only = False
	opt.local_loss = False
	opt.fineSize = 256
	opt.num_folders = 20


	mocker.patch('torch.nn.Linear')
	mocker.patch('models.pix2pix_geo_model.get_downsample', return_value=32)
	gan.initialize(opt)

	torch.nn.Linear.assert_not_called()

	# Check this switch is working
	opt.folder_pred = True

	mocker.patch('torch.nn.Linear')
	mocker.patch('models.pix2pix_geo_model.get_downsample', return_value=32)
	gan.initialize(opt)

	torch.nn.Linear.assert_called_with(2*256**2 / 32**2 * 420*8, 20)

	gan.netG = MockGenerator
	
	gan.netG_DIV = fake_network(mocker, 'fake_DIV')
	gan.netG_Vx = fake_network(mocker, 'fake_Vx')
	gan.netG_Vy = fake_network(mocker, 'fake_Vy')

	fake_dataset = DatasetMock()

	gan.set_input(fake_dataset)
	models.pix2pix_geo_model.get_innermost.assert_called_with(gan.netG, 'UnetSkipConnectionBlock')

	assert(gan.netG.inner_layer == MockInnerLayer)
	assert(MockInnerLayer.register_forward_hook.called)
	MockInnerLayer.register_forward_hook.assert_called_with(models.pix2pix_geo_model.save_output_hook)
	
	gan.folder_fc = fake_network(mocker, 'fake_folder')

	ce_fun_mock = mocker.MagicMock()
	gan.criterionCE = mocker.MagicMock(return_value=ce_fun_mock)

	gan.optimize_parameters(step_no=1)

	print(gan.fake_folder.name)
	assert(gan.fake_folder.name == 'fake_folder_softmax')
	assert(gan.real_folder.name == 'folder_id')

	assert(gan.netD1s[0].call_args_list[0][0][0].name == '[A, mask_float, fake_discrete_output]_detach')
	
	assert(gan.netD1s[0].call_args_list[1][0][0].name 	== ['A', 'mask_float', 'B'])
	
	assert(len(gan.netD1s[0].call_args_list)		== 3)

	assert(gan.netD2s[0].call_args_list[0][0][0].name == '[A, mask_float, fake_DIV, fake_Vx, fake_Vy]_detach')
	
	assert(gan.netD2s[0].call_args_list[1][0][0].name 	== ['A', 'mask_float', 'B_DIV', 'B_Vx', 'B_Vy'])

	assert(len(gan.netD2s[0].call_args_list)		== 3)

	assert(len(ce_fun_mock.call_args_list) == 2)
	assert(ce_fun_mock.call_args_list[1][0][0].name =='fake_folder_softmax')
	assert(ce_fun_mock.call_args_list[1][0][1].name =='folder_id')


def test_exclude_discriminators(basic_gan):
	opt = standard_options()
	opt.num_discrims = 0

	geo = Pix2PixGeoModel()
	geo.initialize(opt)

	assert(len(geo.netD1s) == 0)

	geo.initialize(opt)
	opt.num_discrims = 2

	geo = Pix2PixGeoModel()
	geo.initialize(opt)

	assert(len(geo.netD1s) == 2)


def test_div_only(basic_gan, mocker):
	opt = standard_options()
	opt.num_discrims = 1
	opt.div_only = True

	gan = Pix2PixGeoModel()
	gan.initialize(opt)

	assert(models.networks.define_D.called)
	assert(models.networks.define_D.call_args[0][0] == 5)

	gan.netG_DIV = fake_network(mocker, 'fake_DIV')

	fake_dataset = DatasetMock()

	gan.set_input(fake_dataset)
	gan.optimize_parameters(step_no=1)

	assert(gan.netG.call_args[0][0].name == ['A', 'mask_float'])

	assert(gan.netD2s[0].call_args_list[0][0][0].name == '[A, mask_float, fake_DIV]_detach')
	assert(gan.netD2s[0].call_args_list[1][0][0].name == ['A', 'mask_float', 'B_DIV'])


# def test_gradient_penalty(mocker):

# 	gan = Pix2PixGeoModel()

# 	MockDiscriminator = mocker.MagicMock()

# 	gan.gradient_penalty(MockDiscriminator, torch.Tensor(3), torch.Tensor(0))

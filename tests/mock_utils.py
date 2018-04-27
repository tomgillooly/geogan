try:
	import mock
except ImportError:
	import unittest.mock as mock


class VariableMock(mock.MagicMock):
	def set_name(self, name):
		self.name = name

		return self

	def float(self):
		copy = self.copy()
		copy.name = self.name + '_float'

		return copy

	
	def detach(self):
		copy = self.copy()
		if type(self.name) == list:
			copy.name = '[' + ', '.join(self.name) + ']'
		else:
			copy.name = self.name
		
		copy.name = copy.name + '_detach'

		return copy


def softmax_stub(mock, dim):
	m = mock.copy()
	m.name = mock.name + '_softmax'

	return m


def max_stub(mock, dim, keepdim):
	m = mock.copy()
	m.name += '_max'

	return m


def bce_loss_stub(mock, target, **kwargs):
	m = mock.copy()
	m.name = 'bce_loss_' + m.name

	return m


def nll_loss2d_stub(**kwargs):
	def loss_fun(mock, target, **kwargs):
		m = mock.copy()
		m.name = 'nll_loss_' + m.name

		return m

	return loss_fun


def mse_loss_stub(**kwargs):
	def loss_fun(mock, target, **kwargs):
		m = mock.copy()
		m.name = 'mse_loss_' + m.name

		return m

	return loss_fun


def mean_stub(mocks, dim):
	name_list = []
	for mock in mocks:
		if type(mock.name) == list:
			name_list += mock.name
		else:
			name_list.append(mock.name)
	v = VariableMock()
	v.name = 'mean_(' + ','.join(name_list) + ')'

	return v


class DatasetMock(dict):
	def __getitem__(self, name):
		s = mock.MagicMock(create=True)
		s.name = name

		# print(name, s)

		return s


def zeros_stub(shape):
	m = mock.MagicMock()
	m.name = 'zeros_{}'.format(','.join([str(x) for x in shape]))

	return m


def ones_stub(shape):
	m = mock.MagicMock()
	m.name = 'ones_{}'.format(','.join([str(x) for x in shape]))

	return m


def cat_stub(mocks, dim=None):
	name_list = []
	for mock in mocks:
		# print(mock)
		# print(mock.name)
		if type(mock.name) == list:
			name_list += mock.name
		else:
			name_list.append(mock.name)
	v = VariableMock()
	v.name = name_list

	return v


def variable_stub(mock, **kwargs):
	v = VariableMock().set_name(mock.name)

	return v
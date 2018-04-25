try:
	import mock
except ImportError:
	import unittest.mock as mock


class VariableMock(mock.MagicMock):
	def set_name(self, name):
		self.name = name

		return self

	def float(self):
		self.name += '_float'

		return self


def softmax_stub(mock, dim):
	m = mock.copy()
	m.name += '_softmax'

	return m


def max_stub(mock, dim, keepdim):
	m = mock.copy()
	m.name += '_max'

	return m


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


def cat_stub(mocks, dim):
	return [mock.name for mock in mocks]


def variable_stub(mock):
	v = VariableMock().set_name(mock.name)

	return v
try:
	import mock
except ImportError:
	import unittest.mock as mock


class VariableMock(mock.MagicMock):
	def float(self):
		self.name += '_float'

		return self


def softmax_stub(mock, dim):
	mock.name += '_softmax'

	return mock


def max_stub(mock, dim, keepdim):
	mock.name += '_max'

	return mock


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
	v = VariableMock()
	v.name = mock.name

	return v
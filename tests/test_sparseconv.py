import torch
import pytest

from models.sparse_conv import *

def test_forward_accepts_two_args():
	s = SparseConv2d(3, 3, 3)

	x = torch.randn((1, 3, 9, 9))
	mask = torch.randn((1, 1, 9, 9))
	x.sparse_mask = mask

	s(x)

	s.forward(x)


def test_full_mask_is_same_as_conv2d():
	s = SparseConv2d(1, 1, 2, bias=False)
	c = torch.nn.Conv2d(1, 1, 2, bias=False)

	s.weight.data.fill_(1)
	c.weight.data.fill_(1)

	x = torch.randn(1, 1, 4, 4)
	mask = torch.ones(1, 1, 4, 4)
	x.sparse_mask = mask

	y_s = s(x)
	y_c = c(x)

	assert((y_s == y_c).all())


def test_forward_weights_by_observations():
	s = SparseConv2d(1, 1, 2, bias=False)
	s.weight.data.fill_(1)

	x = torch.randn(1, 1, 4, 4)
	mask = torch.ones(1, 1, 4, 4)

	mask[0, 0, 1:3, 1:3] = 0
	x.sparse_mask = mask

	y = s(x)
	assert(abs(y[0][0][0, 0] - (torch.sum(x[0, 0, 0, 0:2]) + x[0, 0, 1, 0]) * (3 / 4)) < 0.00001)


def test_erodes_mask_with_each_op():
	s = SparseConv2d(1, 1, 2)

	x = torch.randn(1, 1, 6, 6)
	mask = torch.ones(1, 1, 6, 6)
	mask[0, 0, 1:3, 1:3] = 0
	x.sparse_mask = mask

	y = s(x)

	assert(torch.sum(y.sparse_mask[0, 0, 1, 1]) == 0)
	assert(torch.sum(y.sparse_mask) == 24)

def test_can_do_sparse_instancenorm():
	s = SparseInstanceNorm2d(1)

	x = torch.randn(1, 1, 6, 6)
	mask = torch.randn(1, 1, 6, 6)

	x.sparse_mask = mask

	y = s(x)

	assert((y.sparse_mask == x.sparse_mask).all())
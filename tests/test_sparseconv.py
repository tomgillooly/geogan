import torch
import pytest

from models.sparse_conv import *

def test_forward_accepts_two_args():
	s = SparseConv2d(3, 3, 3)

	x = torch.randn((1, 3, 9, 9))
	mask = torch.randn((1, 3, 9, 9))
	x.sparse_mask = mask

	s(x)

	s.forward(x)


# Not sure whether this should be scaled up, but let's do it as expressed in the paper for now
def test_full_mask_is_scaled_conv2d():
	s = SparseConv2d(1, 1, 2, bias=False)
	c = torch.nn.Conv2d(1, 1, 2, bias=False)

	s.weight.data.fill_(1)
	c.weight.data.fill_(1)

	x = torch.randn(1, 1, 4, 4)
	mask = torch.ones(1, 1, 4, 4)
	x.sparse_mask = mask

	y_s = s(x)
	y_c = c(x)

	assert((y_s == y_c / 4).all())


def test_forward_weights_by_observations():
	s = SparseConv2d(1, 1, 2, bias=False)
	s.weight.data.fill_(1)

	x = torch.randn(1, 1, 4, 4)
	mask = torch.ones(1, 1, 4, 4)

	mask[0, 0, 1:3, 1:3] = 0
	x.sparse_mask = mask

	y = s(x)
	assert(abs(y[0][0][0, 0] - (torch.sum(x[0, 0, 0, 0:2]) + x[0, 0, 1, 0]) / 3) < 0.00001)


def test_erodes_mask_with_each_op():
	s = SparseConv2d(1, 1, 2)

	x = torch.randn(1, 1, 6, 6)
	mask = torch.ones(1, 1, 6, 6)
	mask[0, 0, 1:3, 1:3] = 0
	x.sparse_mask = mask

	y = s(x)

	assert(torch.sum(y.sparse_mask[0, 0, 1, 1]) == 0)
	assert(torch.sum(y.sparse_mask) == 24)
	assert(y.shape == (1, 1, 5, 5))
	assert(y.sparse_mask.shape == (1, 1, 5, 5))


def test_can_do_sparse_instancenorm():
	s = SparseInstanceNorm2d(1)

	x = torch.randn(1, 1, 6, 6)
	mask = torch.randn(1, 1, 6, 6)

	x.sparse_mask = mask

	y = s(x)

	assert((y.sparse_mask == x.sparse_mask).all())


def test_combines_mask_across_channels():
	s = SparseConv2d(2, 1, 2)
	s.weight.data.fill_(1)


	x = torch.randn(1, 2, 6, 6)
	mask = torch.ones(1, 2, 6, 6)
	mask[0, 0, 1:3, 1:3] = 0
	x.sparse_mask = mask

	y = s(x)

	expected = (torch.sum(x[0, :, 0, 0:2]).item() + torch.sum(x[0, :, 1, 0]).item() + x[0, 1, 1, 1]) / 7
	assert(abs(y[0, 0, 0, 0] - expected) < 1e-5)

	expected = (torch.sum(x[0, 0, 0, 1:3]).item() + torch.sum(x[0, 1, 0:2, 1:3]).item()) / 6
	assert(abs(y[0, 0, 0, 1] - expected) < 1e-5)

	expected_mask = torch.ones(1, 1, 5, 5)
	expected_mask[0, 0, 1, 1] = 0

	assert((expected_mask == y.sparse_mask).all())


def test_combines_mask_across_channels_multi_output():
	s = SparseConv2d(4, 2, 3)
	s.weight.data.fill_(1)


	x = torch.randn(1, 4, 9, 9)
	mask = torch.ones(1, 4, 9, 9)
	mask[0, 0, 1:4, 1:4] = 0
	x.sparse_mask = mask

	y = s(x)

	expected = (torch.sum(x[0, :, 0:3, 0:3]) - torch.sum(x[0, 0, 1:3, 1:3])) / 32
	assert(abs(y[0, 0, 0, 0] - expected) < 1e-5)

	expected_mask = torch.ones(1, 1, 7, 7)
	expected_mask[0, 0, 1, 1] = 0

	assert((expected_mask == y.sparse_mask).all())

import pytest

def pytest_addoption(parser):
	parser.addoption('--dataroot', action='store', default='~/data/geology')
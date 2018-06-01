from data.geo_pickler import GeoPickler

import os

# dataroot = os.path.expanduser('/storage/Datasets/Geology-NicolasColtice/new_data/test')
# dataroot = os.path.expanduser('~/data/new_geo_data/test')
dataroot = os.path.expanduser('~/data/new_geo_data/validation')

# out_dir = os.path.expanduser('/storage/Datasets/Geology-NicolasColtice/pytorch_records/test')
# out_dir = os.path.expanduser('~/data/geo_data_pkl/test')
out_dir = os.path.expanduser('~/data/geo_data_pkl/validation')

p = GeoPickler(dataroot, out_dir, 256)

p.collect_all()

p.group_by_series()

# p.pickle_all(1000, 100, 10, verbose=True, skip_existing=False)

for i, folder in enumerate(p.folders):
	p.pickle_series(i, 1, 1000, 100, 10)
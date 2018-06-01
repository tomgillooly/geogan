from data.geo_pickler import GeoPickler

import os

dataroot = os.path.expanduser('/storage/Datasets/Geology-NicolasColtice/DS2-1810-RAW-DAT/train')
# dataroot = os.path.expanduser('/storage/Datasets/Geology-NicolasColtice/new_data/test')
# dataroot = os.path.expanduser('~/data/new_geo_data/test')
# dataroot = os.path.expanduser('~/data/new_geo_data/validation')

out_dir = os.path.expanduser('/storage/Datasets/Geology-NicolasColtice/old_pytorch_records/train')
# out_dir = os.path.expanduser('/storage/Datasets/Geology-NicolasColtice/pytorch_records/test')
# out_dir = os.path.expanduser('~/data/geo_data_pkl/test')
# out_dir = os.path.expanduser('~/data/geo_data_pkl/validation')

p = GeoPickler(dataroot, out_dir, 256)

p.collect_all()

p.group_by_series()

groups = [(1, [10, 11, 12, 13, 18, 2, 3, 4, 5, 6, 9]),
			(2, [14, 15, 16, 17, 23]),
			(3, [19, 20, 21, 22])]

thresholds = [0.045, 0.03, 0.03]

thresholds = {str(folder): thresholds[i-1] for i, folders in groups for folder in folders }

# p.pickle_all(thresholds, 100, 10, verbose=True, skip_existing=True)
p.pickle_all(1000, 100, 10, verbose=True, skip_existing=True)

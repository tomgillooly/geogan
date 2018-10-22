import matplotlib.pyplot as plt

import glob
from collections import defaultdict, OrderedDict

legend_lookup = dict((
	('circles_non_filled/test', 'Circles (test)'),
	('circles_non_filled/train', 'Circles (train)'),
	('ellipses3/test', 'Ellipses (test)'),
	('ellipses3/train', 'Ellipses (train)'),
	('old_pytorch_records/test', 'Old geo data'),
	('pytorch_records_new_thresh/train', 'New geo data (train)'),
	('pytorch_records_new_thresh/test', 'New geo data (test)'),
	('voronoi/test', 'Voronoi data (test)'),
	('voronoi/train', 'Voronoi data (train)')
))

files = glob.glob('test_D_results_file_epoch_*')

data_series_min = defaultdict(list)
data_series_max = defaultdict(list)

files = sorted(files,
		key=lambda file: int(file.split('_')[-1]))

epochs = [int(file.split('_')[-1]) for file in files]

for filename in files:
	file = open(filename)

	filetext = file.read()

	data = [line.split(' : ') for line in filetext.splitlines()]

	basedir = '/storage/Datasets/Geology-NicolasColtice/'

	for line in data[0::2]:
		data_series_min[line[0][len(basedir):]].append(float(line[2]))
	
	for line in data[1::2]:
		data_series_max[line[0][len(basedir):]].append(float(line[2]))
	
	# data = [[line[0][len(basedir):],
		# int(line[1]),
		# float(line[2])] for line in data[0::2]]

	# data_min = data[0::2]
	# data_max = data[1::2]

# data_min = [line for line in data if line[1] == 50]
# data_max = [line for line in data if line[1] != 50]

# print(data_min)
# for line in data_max:
# 	plt.scatter(1, line[2])
# 	# plt.text(1.001, line[2], line[0])
# plt.suptitle('data_max')

# plt.legend([line[0] for line in data_max])

# plt.figure()
# for line in data_min:
# 	plt.scatter(1, line[2])
# 	# plt.text(1.001, line[2], line[0])

# plt.legend([line[0] for line in data_min])
# plt.suptitle('data_min')

data_series_min = OrderedDict(sorted(data_series_min.items(), key=lambda item: item[0]))

plt.figure()

for name, series in data_series_min.items():
	if len(name.split('/')) < 3:
		plt.plot(epochs, series, label=legend_lookup[name])

plt.legend()
plt.suptitle('Discriminator output vs epoch (50 data points)')
plt.xlabel("Epoch")
plt.ylabel("D output")


data_series_max = OrderedDict(sorted(data_series_max.items(), key=lambda item: item[0]))

plt.figure()

for name, series in data_series_max.items():
	if len(name.split('/')) < 3:
		plt.plot(epochs, series, label=legend_lookup[name])

plt.legend()
plt.suptitle('Discriminator output vs epoch (maximum available series)')
plt.xlabel("Epoch")
plt.ylabel("D output")

plt.show()
import os
from options.test_options import TestOptions
import matplotlib
matplotlib.use('Agg')
from metrics.hausdorff import get_hausdorff, get_hausdorff_exc
from models.models import create_model
from util.visualizer import Visualizer
from util import html

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io

from data.geo_unpickler import GeoExhaustiveUnpickler
import torch.utils.data
from collections import defaultdict


# Need this if we trained with a different version
import torch._utils
try:
    torch._utils._rebuild_tensor_v2
except AttributeError:
    def _rebuild_tensor_v2(storage, storage_offset, size, stride, requires_grad, backward_hooks):
        tensor = torch._utils._rebuild_tensor(storage, storage_offset, size, stride)
        tensor.requires_grad = requires_grad
        tensor._backward_hooks = backward_hooks
        return tensor
    torch._utils._rebuild_tensor_v2 = _rebuild_tensor_v2


opt = TestOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.batchSize = 1  # test code only supports batchSize = 1
#opt.serial_batches = False  # no shuffle
opt.no_flip = True  # no flip

unpickler = GeoExhaustiveUnpickler()
unpickler.initialise(opt)

dataset = torch.utils.data.DataLoader(
        unpickler,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=opt.nThreads)

dataset_size = len(dataset)
print('#training images = %d' % dataset_size)

model = create_model(opt)

try:
    os.mkdir(web_dir)
except:
    pass

results_file_name = os.path.join(web_dir, opt.name + '_results')

if os.path.exists(results_file_name):
    os.remove(results_file_name)
    
pkl_results_file_name = os.path.join(web_dir, opt.name + '_results.pkl')

if os.path.exists(pkl_results_file_name):
    os.remove(pkl_results_file_name)


metric_data = []

class_labels = ['Ridge', 'Plate', 'Subduction']

results_data = defaultdict(list)

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    
    
    current_metric = model.get_current_metrics()
    metric_data.append(current_metric)
    
    with open(results_file_name, 'a') as results_file:
        results_file.write(', '.join(map(str, current_metric.values())) + '\n')
        
    current_metric['mask_loc'] = (data['mask_x1'], data['mask_y1'])
    current_metric['output'] = model.fake_B_out
    
    results_data[data['series_no']].append(current_metric)

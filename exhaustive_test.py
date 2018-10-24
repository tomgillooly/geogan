import os
from options.test_options import TestOptions
import matplotlib
matplotlib.use('Agg')
from models.models import create_model
from util.visualizer import Visualizer
from util import html

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io

from data.geo_unpickler import GeoExhaustiveUnpickler
import torch.utils.data
from collections import defaultdict

import sqlite3


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
opt.no_flip = True  # no flip

unpickler = GeoExhaustiveUnpickler()
unpickler.initialise(opt)

model = create_model(opt)

results_dir = os.path.join('results', opt.name, 'test_{}'.format(opt.which_epoch))

try:
    os.mkdir(results_dir)
except:
    pass

results_file_name = os.path.join(results_dir, opt.name + '_results')

if os.path.exists(results_file_name):
    os.remove(results_file_name)

results_db = sqlite3.connect('geogan_results.db')

results_c = results_db.cursor()
table_exists_c = results_c.execute('SELECT name FROM sqlite_master WHERE type="table" AND name=?', (opt.name,))

if table_exists_c.fetchone() == None:
    results_c.execute('''CREATE TABLE {} 
        (dataroot text, series int, mask_size int, mask_x int, mask_y int,
        emd_ridge real, emd_subduction real, emd_mean real)'''.format(opt.name))


for i in range(len(unpickler)):
    series_data = unpickler[i]
    
    for data in series_data:
        model.set_input(data)
        for _ in range(opt.test_repeats):
            model.test()
            current_metric = model.get_current_metrics()
            
            with open(results_file_name, 'a') as results_file:
                results_file.write(', '.join(map(str, current_metric.values())) + '\n')
                
            results_c.execute('''INSERT INTO {} VALUES
                (?, ?, ?, ?, ?,
                ?, ?, ?)
                '''.format(opt.name), (opt.dataroot, int(data['series_number']), int(data['mask_size'].numpy()[0]), int(data['mask_x1'].numpy()[0]), int(data['mask_y1'].numpy()[0]),
                    current_metric['EMD_ridge'], current_metric['EMD_subduction'], current_metric['EMD_mean']))
        results_db.commit()

results_db.close()

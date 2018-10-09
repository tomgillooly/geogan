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

from data.geo_unpickler import GeoUnpickler
import torch.utils.data


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

unpickler = GeoUnpickler()
unpickler.initialise(opt)

dataset = torch.utils.data.DataLoader(
        unpickler,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=opt.nThreads)

dataset_size = len(dataset)
print('#training images = %d' % dataset_size)

model = create_model(opt)

web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))

if not opt.no_images:
    visualizer = Visualizer(opt)
    # create website
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
    # test
else:
    try:
        os.mkdir(web_dir)
    except:
        pass

results_file_name = os.path.join(web_dir, opt.name + '_results')

if os.path.exists(results_file_name):
    os.remove(results_file_name)


img_data = []
metric_data = []

class_labels = ['Ridge', 'Plate', 'Subduction']

for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break
    model.set_input(data)
    model.test()
    
    
    current_metric = model.get_current_metrics()
    metric_data.append(current_metric)
    
    with open(results_file_name, 'a') as results_file:
        results_file.write(', '.join(map(str, current_metric.values())) + '\n')
    
    
    if not opt.no_images:
        visuals = model.get_current_visuals()
    
    img_path = model.get_image_paths()

    print('%04d: process image... %s' % (i, img_path))
    # visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

    if not opt.no_images:
        img_data.append((visuals, img_path))


total_metrics = model.accumulate_metrics(metric_data)

text = []
for key, value in total_metrics.items():
        try:
            text.append("{} = {:.04} ".format(key, value))
        except ValueError:
            print('ValueError', key, value)

if text and not opt.no_images:
    webpage.add_text(text)

if opt.no_images:
    for metrics in metric_data:
        text = []

        for key, value in metrics.items():
            try:
                text.append("{} = {:.04} ".format(key, value))
            except ValueError:
                print('ValueError', key, value)
else:
    for (visuals, img_path), metrics in zip(img_data, metric_data):
        

        text = []

        for key, value in metrics.items():
            try:
                text.append("{} = {:.04} ".format(key, value))
            except ValueError:
                print('ValueError', key, value)
        
        if len(visuals) < 6:
            row_lengths = [len(visuals)]
        else:
            row_lengths = [3]
            row_lengths.append(3)
            if opt.continent_data:
                row_lengths.append(1)
            row_lengths.append(2)

        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, row_lengths=row_lengths)

        if text:
            webpage.add_text(text)

if not opt.no_images:
    webpage.save()

import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from metrics.hausdorff import get_hausdorff, get_hausdorff_exc
from models.models import create_model
from util.visualizer import Visualizer
from util import html

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io


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
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)
# create website
web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))
# test

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
    visuals = model.get_current_visuals()
    img_path = model.get_image_paths()

    if opt.visualise_hausdorff:
        mask = data["mask"]

        actual_inpaint_region = data["A"].masked_select(
            mask.repeat(1, 3, 1, 1)).numpy().reshape(3, 100, 100).transpose(1, 2, 0)

        test_inpaint_region = model.fake_B_classes.data.masked_select(
            mask).numpy().reshape(100, 100)


        for c in [0, 2]:
            _, _, _, i_precision, i_recall = get_hausdorff(test_inpaint_region == c, actual_inpaint_region[:, :, c], True)
            gt = np.zeros((actual_inpaint_region.shape[0], actual_inpaint_region.shape[1], 3), dtype=np.uint8)
            gt[:, :, c] = actual_inpaint_region[:, :, c]*255

            inpainted = np.zeros((actual_inpaint_region.shape[0], actual_inpaint_region.shape[1], 3), dtype=np.uint8)
            inpainted[:, :, c] = (test_inpaint_region == c)*255

            # visuals['ground_truth_class_%d' % c] = gt
            # visuals['inpainted_class_%d' % c] = inpainted
            visuals['hausdorff_recall_class_%d' % c] = i_recall
            visuals['hausdorff_precision_class_%d' % c] = i_precision
            
            _, _, _, i_precision, i_recall = get_hausdorff_exc(test_inpaint_region == c, actual_inpaint_region[:, :, c], True)
            visuals['hausdorff_recall_exc_class_%d' % c] = i_recall
            visuals['hausdorff_precision_exc_class_%d' % c] = i_precision
            

    # for key, value in model.get_current_metrics().items():
    #     print(key,"=",value)

    metric_data.append(model.get_current_metrics())

    # if opt.metrics:
    #   thresh = 1000
    #   accuracy = 0

    #   out_im = visuals['fake_B']
    #   # print(np.bincount(out_im.ravel()))
    #   io.imshow(out_im)
    #   io.show()

    print('%04d: process image... %s' % (i, img_path))
    # visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)
    img_data.append((visuals, img_path))


with open(results_file_name, 'a') as results_file:
    total_metrics = model.accumulate_metrics(metric_data)

    text = []
    for key, value in total_metrics.items():
            try:
                text.append("{} = {:.04} ".format(key, value))
            except ValueError:
                print(key, value)

    if text:
        webpage.add_text(text)


    for (visuals, img_path), metrics in zip(img_data, metric_data):

        text = []

        results = []
        for key, value in metrics.items():
            try:
                text.append("{} = {:.04} ".format(key, value))
            except ValueError:
                print(key, value)
            results.append(str(value))

        results_file.write(', '.join(results))
        
        row_lengths = []
        row_lengths.append(4)   # Discrete input, GT, softmax, one-hot output
        row_lengths.append(3)   # Divergence input, output, G
        row_lengths.append(3)   # Velocity x input, output, G
        row_lengths.append(3)   # Velocity y input, output, G
        row_lengths.append(2)   # Class 0 Haussdorf recall, precision
        row_lengths.append(2)   # Class 2 Haussdorf recall, precision
        row_lengths.append(2)   # Class 0 Haussdorf exclusive recall, precision
        row_lengths.append(2)   # Class 2 Haussdorf exclusive recall, precision
        # row_lengths.append(2)   # Class 0 and 2 EM distance


        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio, row_lengths=row_lengths)

        if text:
            webpage.add_text(text)

webpage.save()

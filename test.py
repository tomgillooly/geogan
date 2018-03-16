import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from util import html

import numpy as np
import skimage.io as io

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

with open(results_file_name, 'a') as results_file:
    for i, data in enumerate(dataset):
        if i >= opt.how_many:
            break
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()

        # for key, value in model.get_current_metrics().items():
        #     print(key,"=",value)

        metrics = model.get_current_metrics()

        # if opt.metrics:
        # 	thresh = 1000
        # 	accuracy = 0

        # 	out_im = visuals['fake_B']
        # 	# print(np.bincount(out_im.ravel()))
        # 	io.imshow(out_im)
        # 	io.show()

        print('%04d: process image... %s' % (i, img_path))
        visualizer.save_images(webpage, visuals, img_path, aspect_ratio=opt.aspect_ratio)

        text = []

        results = []
        for key, value in metrics.items():
            try:
                text.append("{} = {:.04} ".format(key, value))
            except ValueError:
                print(key, value)
            results.append(str(value))

        results_file.write(', '.join(results))


        if text:
            webpage.add_text(text)

webpage.save()

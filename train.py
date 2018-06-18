import time
from options.train_options import TrainOptions
from data.geo_unpickler import GeoUnpickler
from models.models import create_model
from util.visualizer import Visualizer

from collections import defaultdict

import glob
import sys

import torch.utils.data

def train():
    opt = TrainOptions().parse()

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
    
    visualizer = Visualizer(opt)
    total_steps = 0

    running_errors = defaultdict(list)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            
            model.set_input(data)
            model.optimize_parameters(step_no=total_steps)
            
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize

            for key, item in model.get_current_errors().items():
                running_errors[key].append(item)

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                average_errors = {}
                for key, error_list in running_errors:
                    average_errors[key] = sum(error_list) / len(error_list)
                
                running_errors = defaultdict(list)

                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, average_errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / (dataset_size * opt.batchSize), opt, average_errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()

if __name__ == '__main__':
    # torch.multiprocessing.set_sharing_strategy("file_system")
    train()
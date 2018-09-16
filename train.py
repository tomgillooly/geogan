import time
from options.train_options import TrainOptions
from data.geo_unpickler import GeoUnpickler
from models.models import create_model
from util.visualizer import Visualizer

from collections import defaultdict

import glob
import sys

import torch.utils.data

from math import ceil

def train():
    opt = TrainOptions().parse()

    # Load in 'list' of data
    unpickler = GeoUnpickler()
    unpickler.initialise(opt)

    # Create shuffler from list of data
    dataset = torch.utils.data.DataLoader(
        unpickler,
        batch_size=opt.batchSize,
        shuffle=not opt.serial_batches,
        num_workers=opt.nThreads)

    dataset_size = len(unpickler)

    # Only do an optimiser step every 10 steps of data
    # This effectively gives a 10x bigger batch size without having to worry
    # about running out of memory
    optimiser_step_interval = 10

    print('#training images = %d' % dataset_size)
    print('#batches = %d' % (len(dataset)/optimiser_step_interval))


    model = create_model(opt)
    
    visualizer = Visualizer(opt)
    total_steps = 0

    # Keep a running average of the errors, instead of printing the error at each epoch
    running_errors = defaultdict(list)

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
        data_iter = iter(dataset)
        
        epoch_start_time = time.time()
        epoch_iter = 0


        # Don't worry about the fractional batch at the end, just keep it simple
        for i in range(int(len(dataset) / optimiser_step_interval)):
            iter_start_time = time.time()
            visualizer.reset()

            model.zero_optimisers()

            # Pass a full batch to either the generator or discriminator
            for j in range(optimiser_step_interval):
                # try-except just to catch end of dataset without failing completely
                try:
                    data = next(data_iter)

                    model.set_input(data)
                    # Doesn't do anything with discriminator, just populates input (conditional), 
                    # target and generated data in object
                    model.forward()

                    model.optimize_G()
                    model.optimize_D()
        
                # Just in case we run off the end of our dataset
                except StopIteration:
                    break
            
            # Update running errors
            for key, item in model.get_current_errors().items():
                running_errors[key].append(item)

            model.step_optimisers()
          
            # Show example images on visualiser
            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            # Print error to console and plot on visualiser
            if total_steps % opt.print_freq == 0:
                average_errors = {}
                for key, error_list in running_errors.items():
                    average_errors[key] = sum(error_list) / len(error_list)
                
                running_errors = defaultdict(list)

                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, average_errors, t)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, average_errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')
             
            total_steps += 1
            epoch_iter += opt.batchSize*optimiser_step_interval


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

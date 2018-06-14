from models.unet.unet_model import UNet
from data.geo_unpickler import GeoUnpickler
from torch.autograd import Variable

import numpy as np
import os
import time
import torch.nn as nn
import torch.utils.data

import util.util as util

import argparse
import visdom

parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, help='name of model')
parser.add_argument('--dataroot', type=str, help='Base directory for data files')
parser.add_argument('--how_many', type=int, default=1, help='Number of images to test on')
parser.add_argument('--which_iter', type=int, default=150, help='Which model iteration to use')
parser.add_argument('--shuffle', action='store_true', help="Don't test the data in order")

def save_im(webpage, image_path, label, im, aspect_ratio=1.0):
	short_path = ntpath.basename(image_path)
	name = os.path.splitext(short_path)[0]

	image_dir = webpage.get_image_dir()

	image_name = '%s_%s.png' % (name, label)
	save_path = os.path.join(image_dir, image_name)
	h, w, _ = im.shape

	if aspect_ratio > 1.0:
		im = imresize(im, (h, int(w * aspect_ratio)), interp='bicubic')
	if aspect_ratio < 1.0:
		im = imresize(im, (int(h / aspect_ratio), w), interp='bicubic')
	plt.imsave(save_path, im)

	return image_name, label, image_name

def main(opt):
	print('Training ' + opt.name)
	dataroot = opt.dataroot

	u = GeoUnpickler(dataroot)
	u.collect_all()

	norm_layer = torch.nn.InstanceNorm2d
	if opt.norm == 'batch':
		norm_layer = torch.nn.BatchNorm2d

	l = UNet(3, 1, norm_layer)

	print(l)

	save_filename = '{}_net_labeller.pth'.format(opt.which_iter)
	save_path = os.path.join('checkpoints', opt.name, save_filename)
	c.load_state_dict(torch.load(save_path))

	if torch.cuda.is_available():
		l.cuda(0)

	dataset = torch.utils.data.DataLoader(
		u,
		shuffle=opt.shuffle,
		batch_size=1,
		num_workers=2)

	print('dataset length = ' + str(len(dataset)))


	webpage = html.HTML(os.path.join('results', opt.name), 'Component area prediction', reflesh=1)

	webpage.add_header('Component area prediction')

	for i, data in enumerate(dataset):
		if i >= opt.how_many:
			break

		input = Variable(data['A'])
		input = input.cuda(0) if torch.cuda.is_available() else input

		y_hat = l(input)

		y = Variable(data['area_img'], requires_grad=False)
		y = y.cuda(0) if torch.cuda.is_available() else y

		l1_loss = loss_fn(y_hat.squeeze(), y.float())

		loss = l1_loss


		image_path = 'serie_{}_{:05}'.format(data['folder_id'][0], data['series_number'][0])
		print('Processing [{}] - {}'.format(i, image_path))
		real_DISC = data['B'].numpy().squeeze().transpose(1, 2, 0)
		real_DISC = np.interp(real_DISC, [np.min(real_DISC), np.max(real_DISC)], [0, 255]).astype(np.uint8)

		visuals = []

		visuals.append(save_im(webpage, image_path, 'input_discrete', real_DISC))

		visuals.append(save_plot(webpage, image_path, 'ground_truth', data['area_img'].numpy()))
		visuals.append(save_plot(webpage, image_path, 'predicted', y_hat.data.numpy()))

		image_names, labels, _ = zip(*visuals)
		
		webpage.add_header(image_path)
		webpage.add_images(image_names, labels, image_names)
		webpage.add_text(['L1 loss = {:03}'.format(loss.data[0])])

	webpage.save()



if __name__ == '__main__':
	options = parser.parse_args()
	
	if not os.path.exists(os.path.join('checkpoints', options.name)):
		os.mkdir(os.path.join('checkpoints', options.name))

	
	main(options)
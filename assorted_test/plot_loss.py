#!/usr/bin/env python3

import glob
import os
import re

from collections import defaultdict, namedtuple
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.colors as colors
import matplotlib.cm as cmx

import sys

NameOrder = namedtuple('NameOrder', ['name', 'order'])

better_legend_lookup = {
	"D1_fake": NameOrder("D1(fake)", 1),				# Its not the average, just the last in the iteration
	"D_fake": NameOrder("D(fake)", 1),				# Its not the average, just the last in the iteration
	"D1_real": NameOrder("D1(real)", 2),
	"D_real": NameOrder("D(real)", 2),
	"D2_fake": NameOrder("D2(fake)", 3),
	"D2_real": NameOrder("D2(fake)", 4),
	"D_grad_pen": NameOrder("D gradient penalty", 5),
	"D1_grad_pen": NameOrder("D1 gradient penalty", 6),
	"D2_grad_pen": NameOrder("D2 gradient penalty", 7),
	"G_GAN_D": NameOrder("-D(fake) (G training)", 8),
	"G_GAN_D1": NameOrder("-D1(fake) (G training)", 8),
	"G_GAN_D2": NameOrder("-D2(fake) (G training)", 9),
	"G_L2": NameOrder("Continuous L2", 10),
	"G_L2_DIV": NameOrder("Continuous L2", 10),
	"G_CE": NameOrder("Discrete CE", 11),
	"G": NameOrder("G (Adversarial + CE + L2)", 12),
	"folder_CE": NameOrder("Folder prediction CE", 13),
	"D": NameOrder("Critic loss", 14),
	"G_L2_grad_y": NameOrder("L2 Y gradient", 15),
	"G_L2_grad_x": NameOrder("L2 X gradient", 16),
}


# fig=plt.figure(figsize=(12, 10))
# ax = [plt.gca()]
# [ax.append(fig.add_subplot(210+i)) for i in range(1, 3)]

assert(len(sys.argv) > 1)

models = sys.argv[1:]

extra_artists = []

# with open('checkpoints/geo_pix2pix_wgan_multiple_critic/loss_log.txt') as file:
# for j, filename in enumerate(['../checkpoints/archive/only_old_data/{}/loss_log.txt'.format(model) for model in 'base_autoencoder', 'base_autoencoder_wce', 'geo_pix2pix_wgan_base', 'geo_pix2pix_wgan_weighted_ce']):
# for j, model_name in enumerate(['div_inline_ae_weighted_folder_pred', 'div_inline_ae_big_batch_new_thresh', 'div_inline_ae_continents', 'div_inline_wgan_base', 'div_inline_ae_big_batch_new_thresh_weighted']):
# for j, model_name in enumerate(['critic_capacity_pre_train_base', 'critic_capacity_rand_init']):
for j, model_name in enumerate(models):
# for filename in glob.glob('checkpoints/*/loss_log.txt'):
	filename = 'checkpoints/{}/loss_log.txt'.format(model_name)
	# model_name = filename.split('/')[2]
	
	fig=plt.figure(figsize=(12, 10))
	ax = plt.gca()
	# if os.path.exists(os.path.join('results', model_name, 'loss_plot.svg')):
	# 	continue

	print(model_name)

	with open(filename) as file:

		epochs = []
		iters = []
		iters
		G = []
		G_GAN_D1 = []
		G_GAN_D2 = []
		G_L2 = []
		G_CE = []
		D1_real = []
		D1_fake = []
		D2_real = []
		D2_fake = []

		plot_data = defaultdict(list)
		iter_data = defaultdict(list)

		iter_re = re.compile("\((\w+): (\d+), (\w+): (\d+), (\w+): ([-\d\.]+)\)")
		plot_re = re.compile("(\w+): ([-\d\.]+)")

		for line in file:
			iter_match = iter_re.match(line)
			
			if not iter_match:
				continue

			iter_items = iter_match.groups()
			plot_items = plot_re.findall(line[iter_match.end():])
			
			for key, value in [(plot_item[0], float(plot_item[1])) for plot_item in plot_items]:
				plot_data[key].append(value)
			
			for key, value in [(iter_items[i], float(iter_items[i+1])) for i in range(0, len(iter_items), 2)]:
				iter_data[key].append(value)
			
		if not iter_data or not plot_data:
			continue

		dtype = [(key, float) for key in iter_data.keys()]
		
		# Sort data by epoch/iter
		values = list(zip(*[iter_data[k] for k in iter_data.keys()]))
		iter_np = np.array(values, dtype=dtype)
		sort_idx = np.argsort(iter_np, order=['epoch', 'iters'])
		for key in plot_data.keys():
			plot_data[key] = [plot_data[key][idx] for idx in sort_idx]

		first_epoch = np.min(iter_data['epoch'])
		last_epoch = np.max(iter_data['epoch'])

		x_data = np.linspace(first_epoch, last_epoch+1, len(iter_data['epoch']))

		ordered_keys = sorted(plot_data.keys(), key=lambda key: better_legend_lookup[key].order)
		
		# y_data = np.array([plot_data[k] for k in ordered_keys]).squeeze()

		# plt.plot(np.stack([x_data] * len(plot_data.keys())).T,
		# 	y_data.T, cmap='viridis')

		cm = plt.get_cmap('rainbow')
		cNorm  = colors.Normalize(vmin=0, vmax=len(ordered_keys))
		scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

		for i, k in enumerate(ordered_keys):
			colorVal = scalarMap.to_rgba(i)

			ax.plot(x_data.T, np.array(plot_data[k]).T, color=colorVal)
		
		ax.set_xlim([first_epoch-0.1*(last_epoch-first_epoch+1), last_epoch+1])		
		# plt.plot(y_data.T)

		# if j % 2 == 1:
			# pos = ax[j].get_position()
			# location_xy = pos.x+pos.w+10, pos.y+pos.h
		lgd = ax.legend([better_legend_lookup[key].name for key in ordered_keys], bbox_to_anchor=(1.02, 0, 1.12, 1), loc=2, borderaxespad=0.)

		extra_artists.append(lgd)

		ax.set_title(model_name)
		ax.grid()

# plt.tight_layout(rect=(0, 0, 1.15, 1))		
	plt.savefig('results/{}/loss_plot.png'.format(model_name), dpi=90,
		additional_artists=extra_artists, bbox_inches='tight')
	# plt.show()

# plt.savefig('../results/' + model_name + '/loss_plot.svg', dpi=90)

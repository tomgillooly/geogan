#!/usr/env/bin python

import glob
import numpy as np
import os
import subprocess
import sys
import random
import re

data_desc_lookup = {
	'circles': 'Filled circles, no more than 1 circle removed by mask, at most 50\% of area remains',
	'circles_non_filled_mask_loc': 'Non-filled circles, at least 10 pixels guaranteed in mask region',
	'circles_non_filled_mixed': 'Non-filled circles, 50\% with at least 10 pixels guaranteed in mask region, 50\% with nothing in mask region',
	'circles_non_filled': 'First 10000 guaranteed maximum 1 circle missing, 10 pixels minimum. Remaining 90000, masks placed arbitrarily',
	'ellipses': 'Ten pixels minimum in masked region, ellipses with major/minor aes ranging from 3 to 20 pixels',
	'ellipses_overfit': 'First 1000 data points from ellipses set, for testing whether a model can overfit'
}

# assert(len(sys.argv[1:]))

# models = sys.argv[1:]

models = open('models_to_test').read().split('\n')
models = models


weight_fun_re = re.compile('diff_in_numerator: (\w+)')
local_loss_re = re.compile('local_loss: (\w+)')
grad_loss_re = re.compile('grad_loss: (\w+)')
weighted_grad_re = re.compile('weighted_grad: (\w+)')
num_discrims_re = re.compile('num_discrims: (\d+)')
batch_size_re = re.compile('batchSize: (\d+)')
learning_rate_re = re.compile('lr: ([\d\.e-]+)')
dataroot_re = re.compile('dataroot: ([\w/-]+)')


with open('circle_results_report.tex', 'w') as out_file:
	out_file.write('\n'.join([r'\documentclass{article}',
		r'\usepackage{graphicx}',
		r'\usepackage{subcaption}',
		r'\usepackage{amsmath}',
		r'\usepackage[table]{xcolor}',
		'\\usepackage{geometry}\n'
			 '\\geometry{\n'
			 'a4paper,\n'
			 'total={170mm,257mm},\n'
			 'left=20mm,\n'
			 'top=20mm,\n'
			 '}\n',
		'\\begin{document}',
		'\\tableofcontents\n'
		]))
	out_file.write('\n')

	out_file.write('\n'.join(
		[
			r'\section{Conventions}',

			'The dataset description refers to the training dataset. All tests are conducted on a dataset where the mask region has been placed arbitrarily\n',

			'$F_k$ refers to pixels of a class $k$ in the ground truth image.\n',

			"$F_{k'}$ refers to pixels of a class $k$ in the output image.\n",

			'$T$ refers to the total number of pixels\n',

			'Global loss means L2 loss is taken on the whole image, local loss is just within the mask region\n',
			
			'Gradient loss is L2 loss on the gradient images, and the region of interest is the same as the overall L2 loss (i.e. global or local)\n',

			'Reported L2 error is averaged over 1000 images\n'
			]
		))
	out_file.write('\\clearpage\n')

	model_params = [{} for _ in range(len(models))]
	
	smallest_l2_global = np.Inf
	smallest_l2_global_idx = -1

	smallest_l2_local = np.Inf
	smallest_l2_local_idx = -1

	biggest_l2_global = -np.Inf
	biggest_l2_global_idx = -1

	biggest_l2_local = -np.Inf
	biggest_l2_local_idx = -1

	for j, model in enumerate(models):
		results_text = open(os.path.join('results', model, 'test_latest', '{}_results'.format(model))).read()

		global_losses, local_losses = zip(*[[float(y) for y in x.split(', ')] for x in results_text.splitlines()])
		model_params[j]['l2_loss_global'] = np.mean(global_losses)
		model_params[j]['l2_loss_local'] = np.mean(local_losses)
		
		if model_params[j]['l2_loss_global'] < smallest_l2_global:
			smallest_l2_global = model_params[j]['l2_loss_global']
			smallest_l2_global_idx = j

		if model_params[j]['l2_loss_local'] < smallest_l2_local:
			smallest_l2_local = model_params[j]['l2_loss_local']
			smallest_l2_local_idx = j
		
		if model_params[j]['l2_loss_global'] > biggest_l2_global:
			biggest_l2_global = model_params[j]['l2_loss_global']
			biggest_l2_global_idx = j

		if model_params[j]['l2_loss_local'] > biggest_l2_local:
			biggest_l2_local = model_params[j]['l2_loss_local']
			biggest_l2_local_idx = j

	out_file.write('\\begin{table}[h]\n')
	# out_file.write('\\begin{tabular}{|p{5cm}|p{10cm}|}\n')
	out_file.write('\\centering\n')
	out_file.write('\\begin{tabular}{|p{2.2cm}|p{1cm}|p{1cm}|p{1.5cm}|p{1.5cm}|p{1.5cm}|p{1cm}|p{1.5cm}|p{1.5cm}|}\n')
	out_file.write('\t\\hline\n')
	out_file.write(' & Batch Size & Loss & Learning Rate & Weight Function & Gradient Loss & Critic & L2 (Global) & L2 (Local) \\\\\n')
	out_file.write('\t\\hline\n')


	for j, model in enumerate(models):
		print(model)
		print(j)

		# options = open(os.path.join('checkpoints', model, 'opt.txt')).read()
		slurm_files = sorted(glob.glob(os.path.join('checkpoints', model, 'slurm*')))
		options = open(slurm_files[0]).read()

		results_text = open(os.path.join('results', model, 'test_latest', '{}_results'.format(model))).read()

		global_losses, local_losses = zip(*[[float(y) for y in x.split(', ')] for x in results_text.splitlines()])
		model_params[j]['l2_loss_global'] = np.mean(global_losses)
		model_params[j]['l2_loss_local'] = np.mean(local_losses)
		
		if model_params[j]['l2_loss_global'] < smallest_l2_global:
			smallest_l2_global = model_params[j]['l2_loss_global']
			smallest_l2_global_idx = j

		if model_params[j]['l2_loss_local'] < smallest_l2_local:
			smallest_l2_local = model_params[j]['l2_loss_local']
			smallest_l2_local_idx = j
		
		if model_params[j]['l2_loss_global'] > biggest_l2_global:
			biggest_l2_global = model_params[j]['l2_loss_global']
			biggest_l2_global_idx = j

		if model_params[j]['l2_loss_local'] > biggest_l2_local:
			biggest_l2_local = model_params[j]['l2_loss_local']
			biggest_l2_local_idx = j

		# print(options)

		weight_fun_match = weight_fun_re.search(options)

		weight_fun_match = weight_fun_re.search(options)
		if weight_fun_match:
			diff_in_numerator = weight_fun_match.group(1) == 'True'
			model_params[j]['weight_fn'] = ''.join(['$\\frac', '{2 + \\lvert F_k - F_{k\'} \\rvert}' if diff_in_numerator else '{2}', '{F_k + F_{k\'}}$'])
		else:
			model_params[j]['weight_fn'] = ''.join(['$\\frac', '{\\lvert T \\rvert}', '{\\lvert {F_k} \\rvert}$'])

		model_params[j]['local_loss'] = local_loss_re.search(options).group(1) == 'True'
		model_params[j]['batch_size'] = int(batch_size_re.search(options).group(1))*10
		model_params[j]['learning_rate'] = float(learning_rate_re.search(options).group(1))
		model_params[j]['dataroot'] = dataroot_re.search(options).group(1)
		model_params[j]['grad_loss'] = grad_loss_re.search(options).group(1) == 'True' if grad_loss_re.search(options) else False
		model_params[j]['weighted_grad'] = weighted_grad_re.search(options).group(1) == 'True' if weighted_grad_re.search(options) else False
		model_params[j]['critic'] = int(num_discrims_re.search(options).group(1)) > 0

		out_file.write('\tExperiment {} & {} & {} & {} & {} & {} & {} & {}{:0.4} & {}{:0.4} \\\\\n'.format(
			j+1,
			model_params[j]['batch_size'],
			'Local' if model_params[j]['local_loss'] else 'Global',
			model_params[j]['learning_rate'],
			model_params[j]['weight_fn'],
			'No' if  not model_params[j]['grad_loss'] else 'Yes' if not model_params[j]['weighted_grad'] else 'Weighted',
			'Yes' if model_params[j]['critic'] else 'No',
			'\\cellcolor{red}' if j == biggest_l2_global_idx else '\\cellcolor{green}' if j == smallest_l2_global_idx else '',
			model_params[j]['l2_loss_global'],
			'\\cellcolor{red}' if j == biggest_l2_local_idx else '\\cellcolor{green}' if j == smallest_l2_local_idx else '',
			model_params[j]['l2_loss_local']
			))
		out_file.write('\t\\hline\n')


	out_file.write('\\end{tabular}\n')
	out_file.write('\\end{table}\n')
		
	out_file.write('\\clearpage\n')

	for j, model in enumerate(models):
		options = open(os.path.join('checkpoints', model, 'opt.txt')).read()

		# num_training_data_images = int(subprocess.check_output(
		# 	['ssh', 'oggy', 'find {} -name *.pkl | wc'.format(dataroot)]
		# 	).split()[0])

		last_slurm_file = sorted(glob.glob(os.path.join('checkpoints', model, 'slurm*')))[0]
		num_batches_re = re.compile('#training images = (\d+)')
		num_batches_match = num_batches_re.search(open(last_slurm_file).read())
		num_training_data_images = int(num_batches_match.group(1)) * model_params[j]['batch_size']/10


		# out_file.write('\\section{%s}\n' % ' '.join(model.split('_')))
		out_file.write('\\section{Experiment %d}\n' % (j+1))
		
		# out_file.write('model name : {}'.format('\\_'.join(model.split('_'))))
		
		out_file.write('\\begin{table}[h]\n')
		out_file.write('\\begin{tabular}{|p{5cm}|p{10cm}|}\n')
		out_file.write('\t\\hline\n')
		out_file.write('\tWeighting function & {} \\\\\n'.format(model_params[j]['weight_fn']))
		out_file.write('\t\\hline\n')
		out_file.write('\tLoss & {} \\\\\n'.format('Local' if model_params[j]['local_loss'] else 'Global'))
		out_file.write('\t\\hline\n')
		out_file.write('\tGradient Loss & {} \\\\\n'.format('No' if  not model_params[j]['grad_loss'] else 'Yes' if not model_params[j]['weighted_grad'] else 'Weighted'))
		out_file.write('\t\\hline\n')
		out_file.write('\tNum training images & {} \\\\\n'.format(num_training_data_images))
		out_file.write('\t\\hline\n')
		out_file.write('\tBatch size & {} \\\\\n'.format(model_params[j]['batch_size']))
		out_file.write('\t\\hline\n')
		out_file.write('\tLearning rate & {} \\\\\n'.format(model_params[j]['learning_rate']))
		out_file.write('\t\\hline\n')
		out_file.write('\tDataset description & {} \\\\\n'.format(data_desc_lookup[os.path.basename(model_params[j]['dataroot'])]))
		out_file.write('\t\\hline\n')

		out_file.write('\\end{tabular}\n')
		out_file.write('\\end{table}\n')

		if os.path.exists(os.path.join('results', model, 'remarks')):
			out_file.write('\\subsection{Remarks}')
			
			with open(os.path.join('results', model, 'remarks')) as remark_file:
				remark_text = remark_file.read()

				for line in remark_text.splitlines():
					out_file.write(line + '\n')
		
		im_base_dir = os.path.join('results', model, 'test_latest', 'images')

		all_images = glob.glob(os.path.join(im_base_dir, '*.png'))
		all_images = [os.path.basename(filename) for filename in all_images]

		series_numbers = [int(filename[len('serie_'):len('serie_')+1] + filename[len('serie_x_'):len('serie_x_')+5]) for filename in all_images]
		series_numbers = list(set(series_numbers))

		series_numbers = random.sample(series_numbers, 10)

		
		for series_no in series_numbers:
			filename_base = 'serie_{}_{:05}'.format(series_no / 10000, series_no % 10000)
			
			out_file.write('\\begin{figure}[h]\n')
			
			out_file.write('\t\\begin{subfigure}{0.3\\linewidth}\n')
			out_file.write('\t\\includegraphics[width=\\linewidth]{%s}\n' % os.path.join(im_base_dir, '{}_input_one_hot.png'.format(filename_base)))
			out_file.write('\t\\subcaption{Input discrete}')
			out_file.write('\t\\end{subfigure}\n')
			out_file.write('\t\\begin{subfigure}{0.3\\linewidth}\n')
			out_file.write('\t\\includegraphics[width=\\linewidth]{%s}\n' % os.path.join(im_base_dir, '{}_ground_truth_divergence.png'.format(filename_base)))
			out_file.write('\t\\subcaption{GT divergence}')
			out_file.write('\t\\end{subfigure}\n')
			out_file.write('\t\\begin{subfigure}{0.3\\linewidth}\n')
			out_file.write('\t\\includegraphics[width=\\linewidth]{%s}\n' % os.path.join(im_base_dir, '{}_output_divergence.png'.format(filename_base)))
			out_file.write('\t\\subcaption{Output divergence}')
			out_file.write('\t\\end{subfigure}\n')
			
			out_file.write('\t\\caption{%s}\n' % ' '.join(filename_base.split('_')))
			out_file.write('\\end{figure}\n')
			out_file.write('\n')

		out_file.write('\\begin{figure}[h]\n')
		out_file.write('\t\\includegraphics[width=\\linewidth]{%s}\n' % os.path.join('results', model, 'loss_plot.png'.format(filename_base)))
		out_file.write('\\end{figure}\n')
		out_file.write('\n')


		out_file.write('\\clearpage\n')



		# print(batch_size)
		# print(dataroot)
		# print(num_training_data_images)
		# print(local_loss)
		# print(learning_rate)



	out_file.write(r'\end{document}')
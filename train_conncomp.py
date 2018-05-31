from models.conncomp_net import ConnCompNet
from data.geo_unpickler import GeoUnpickler
from torch.autograd import Variable

import os
import torch.nn as nn
import torch.utils.data

import argparse
import visdom

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default=os.path.expanduser('~/data/geo_data_pkl/test'), help='Base directory for data files')
parser.add_argument('--num_epochs', type=int, default=200, help='Number of training epochs')
parser.add_argument('--display_port', type=int, default=-1, help='Port for visdom')
parser.add_argument('--batch_size', type=int, default=32, help='Number of samples in batch')


def main(opt):
	dataroot = opt.dataroot

	u = GeoUnpickler(dataroot)
	u.collect_all()

	im_size = u[0]['A'].numpy().transpose(1, 2, 0).shape
	hist_size = u[0]['conn_comp_hist'].numpy().shape[0]

	c = ConnCompNet(im_size, hist_size)
	if torch.cuda.is_available():
		c.cuda(1)

	dataset = torch.utils.data.DataLoader(
		u,
		shuffle=True,
		batch_size=opt.batch_size,
		num_workers=2)

	loss_fn = nn.MSELoss()
	optimiser = torch.optim.Adam(c.parameters(), lr=0.0002, betas=(0.5, 0.999))

	if opt.display_port != -1:
		vis = visdom.Visdom(port=opt.display_port)

	total_steps = 0

	loss_x = []
	loss_y = []

	for epoch in range(opt.num_epochs):
		for i, data in enumerate(dataset):
			total_steps += 1

			input = Variable(data['B'])
			input = input.cuda(1) if torch.cuda.is_available() else input

			y_hat = c()

			y = Variable(data['conn_comp_hist'], requires_grad=False)
			y = y.cuda(1) if torch.cuda.is_available() else y

			loss = loss_fn(y_hat, y)

			optimiser.zero_grad()

			loss.backward()

			optimiser.step()

			if total_steps % 100 == 0:
				loss_x.append(total_steps * 1.0 / len(dataset))
				loss_y.append(loss.data)

				if opt.display_port != -1:
					vis.line(loss_x, loss_y,
						options={
						'title': 'Conn commp L2 histogram loss'
						}, win=0)

		if epoch % 10 == 0:
			save_filename = '%s_net_conncomp.pth' % (str(epoch))
	        save_path = os.path.join('results', 'conncomp', save_filename)
	        torch.save(c.cpu().state_dict(), save_path)
	        if torch.cuda.is_available():
	            c.cuda(1)


if __name__ == '__main__':
	options = parser.parse_args()

	main(options)
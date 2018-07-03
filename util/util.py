from __future__ import print_function
import torch
import numpy as np
from PIL import Image
import os


# Converts a Tensor into a Numpy array
# |imtype|: the desired type of the converted numpy array
def tensor2im(image_tensor, imtype=np.uint8):
    image_numpy = image_tensor[0].cpu().float().numpy().squeeze()
    if len(image_numpy.shape) == 2:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    # image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    image_numpy = np.transpose(image_numpy, (1, 2, 0))
    image_numpy = np.interp(image_numpy, [np.min(image_numpy), np.max(image_numpy)], [0, 255])

    return image_numpy.astype(imtype)


def diagnose_network(net, name='network'):
    mean = 0.0
    count = 0
    for param in net.parameters():
        if param.grad is not None:
            mean += torch.mean(torch.abs(param.grad.data))
            count += 1
    if count > 0:
        mean = mean / count
    print(name)
    print(mean)


def save_image(image_numpy, image_path):
    image_pil = Image.fromarray(image_numpy)
    image_pil.save(image_path)


def print_numpy(x, val=True, shp=False):
    x = x.astype(np.float64)
    if shp:
        print('shape,', x.shape)
    if val:
        x = x.flatten()
        print('mean = %3.3f, min = %3.3f, max = %3.3f, median = %3.3f, std=%3.3f' % (
            np.mean(x), np.min(x), np.max(x), np.median(x), np.std(x)))


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def create_weight_mask(tensor1, tensor2, diff_in_numerator=False, method='freq'):
    assert(tensor1.shape == tensor2.shape)

    if method == 'freq':
        total_pixels = tensor1[0, 0, :, :].numel()

        num_pixels1 = tensor1.sum(2, keepdim=True).sum(3, keepdim=True).float()
        num_pixels2 = tensor2.sum(2, keepdim=True).sum(3, keepdim=True).float()

        pixel_freq1 = num_pixels1 / total_pixels
        pixel_freq2 = num_pixels2 / total_pixels

        if diff_in_numerator:
            pixel_weights = (2.0 + torch.abs(pixel_freq1 - pixel_freq2)) / (pixel_freq1 + pixel_freq2)
        else:
            pixel_weights = 2.0 / (pixel_freq1 + pixel_freq2)

        pixel_weights /= torch.sum(pixel_weights + 1e-8, dim=1, keepdim=True)

        if tensor1.device.type != 'cpu':
            pixel_weights = pixel_weights.cuda()

        weight_mask = torch.max(pixel_weights * torch.max(tensor1, tensor2), dim=1, keepdim=True)[0]

    else:
        gt_fg = torch.max(tensor1[:, [0, 2], :, :], dim=1, keepdim=True)[0]
        output_fg = torch.max(tensor2[:, [0, 2], :, :], dim=1, keepdim=True)[0]

        intersection = torch.min(torch.cat((gt_fg.unsqueeze(4), output_fg.unsqueeze(4)), dim=4), dim=4)[0]
        union = torch.max(torch.cat((gt_fg.unsqueeze(4), output_fg.unsqueeze(4)), dim=4), dim=4)[0]

        union_sum = union.sum(3).sum(2)
        iou = intersection.sum(3).sum(2) / union_sum
        iou = 1.0 - iou.unsqueeze(2).unsqueeze(3)
        weight_mask = torch.max(iou * union, dim=1, keepdim=True)[0]
    
    return weight_mask



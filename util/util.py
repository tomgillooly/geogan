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

        num_ridge_pixels_1 = torch.sum(torch.sum(tensor1[:, 0, :, :].unsqueeze(1),
            dim=2, keepdim=True), dim=3, keepdim=True).float()
        num_plate_pixels_1 = torch.sum(torch.sum(tensor1[:, 1, :, :].unsqueeze(1),
            dim=2, keepdim=True), dim=3, keepdim=True).float()
        num_subduction_pixels_1 = torch.sum(torch.sum(tensor1[:, 2, :, :].unsqueeze(1),
            dim=2, keepdim=True), dim=3, keepdim=True).float()

        num_ridge_pixels_2 = torch.sum(torch.sum(tensor2[:, 0, :, :].unsqueeze(1),
            dim=2, keepdim=True), dim=3, keepdim=True).float()
        num_plate_pixels_2 = torch.sum(torch.sum(tensor2[:, 1, :, :].unsqueeze(1),
            dim=2, keepdim=True), dim=3, keepdim=True).float()
        num_subduction_pixels_2 = torch.sum(torch.sum(tensor2[:, 2, :, :].unsqueeze(1),
            dim=2, keepdim=True), dim=3, keepdim=True).float()

        ridge_freq_1 = num_ridge_pixels_1 / total_pixels
        plate_freq_1 = num_plate_pixels_1 / total_pixels
        subduction_freq_1 = num_subduction_pixels_1 / total_pixels

        ridge_freq_2 = num_ridge_pixels_2 / total_pixels
        plate_freq_2 = num_plate_pixels_2 / total_pixels
        subduction_freq_2 = num_subduction_pixels_2 / total_pixels

        if diff_in_numerator:
            ridge_weight = 2.0 + torch.abs(ridge_freq_1 - ridge_freq_2) / (ridge_freq_1 + ridge_freq_2)
            plate_weight = 2.0 + torch.abs(plate_freq_1 - plate_freq_2) / (plate_freq_1 + plate_freq_2)
            subduction_weight = 2.0 + torch.abs(subduction_freq_1 - subduction_freq_2) / (subduction_freq_1 + subduction_freq_2)
        else:
            ridge_weight = 2 * 1.0 / (ridge_freq_1 + ridge_freq_2)
            plate_weight = 2 * 1.0 / (plate_freq_1 + plate_freq_2)
            subduction_weight = 2 * 1.0 / (subduction_freq_1 + subduction_freq_2)

        pixel_weights = torch.cat((ridge_weight, plate_weight, subduction_weight), dim=1)
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



"""

Modified from
Main Author of this file: TATrack
Repo: https://github.com/hekaijie123/TATrack/tree/main
File: https://github.com/hekaijie123/TATrack/blob/main/videoanalyst/data/target/target_impl/utils/make_densebox_target.py

"""

import math
import torch

def gauss_1d(sz, sigma, center, end_pad=0, density=False) -> torch.Tensor:
    k = torch.arange(-(sz - 1) / 2, (sz + 1) / 2 + end_pad).reshape(1, -1)
    gauss = torch.exp(-1.0 / (2 * sigma ** 2) * (k - center.reshape(-1, 1)) ** 2)
    if density:
        gauss /= math.sqrt(2 * math.pi) * sigma
    return gauss


def gauss_2d(sz, sigma, center, end_pad=(0, 0), density=False) -> torch.Tensor:
    if isinstance(sigma, (float, int)):
        sigma = (sigma, sigma)
    return gauss_1d(sz[0].item(), sigma[0], center[:, 0], end_pad[0], density).reshape(center.shape[0], 1, -1) * \
           gauss_1d(sz[1].item(), sigma[1], center[:, 1], end_pad[1], density).reshape(center.shape[0], -1, 1)


def gaussian_label_function(target_bb, sigma_factor=0.1, kernel_sz=1, feat_sz=16, image_sz=256, end_pad_if_even=True, density=False, uni_bias=0) -> torch.Tensor:
    """Construct Gaussian label function.
    target_bb: [b x [x1,y1,x2,y2]]
    
    """

    if isinstance(kernel_sz, (float, int)):
        kernel_sz = (kernel_sz, kernel_sz)
    if isinstance(feat_sz, (float, int)):
        feat_sz = (feat_sz, feat_sz)
    if isinstance(image_sz, (float, int)):
        image_sz = (image_sz, image_sz)

    image_sz = torch.Tensor(image_sz)
    feat_sz = torch.Tensor(feat_sz)

    target_center = (target_bb[:, 0:2] +target_bb[:, 2:4]) * 0.5 
    target_center_norm = (target_center - image_sz / 2) / image_sz

    center = feat_sz * target_center_norm + 0.5 * \
             torch.Tensor([(kernel_sz[0] + 1) % 2, (kernel_sz[1] + 1) % 2])

    sigma = sigma_factor * feat_sz.prod().sqrt().item()

    if end_pad_if_even:
        end_pad = (int(kernel_sz[0] % 2 == 0), int(kernel_sz[1] % 2 == 0))
    else:
        end_pad = (0, 0)

    gauss_label = gauss_2d(feat_sz, sigma, center, end_pad, density=density)
    if density:
        sz = (feat_sz + torch.Tensor(end_pad)).prod()
        label = (1.0 - uni_bias) * gauss_label + uni_bias / sz
    else:
        label = gauss_label + uni_bias
    return label


label_1 = gaussian_label_function( torch.tensor([54, 54, 112, 112]).view(1,-1))
label_2 = gaussian_label_function( torch.tensor([74, 74, 132, 132]).view(1,-1))

label = label_2-(0.5*label_1)
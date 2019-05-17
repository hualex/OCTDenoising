import numpy as np
import matplotlib.pyplot as plt
import skimage

from scipy.ndimage.filters import convolve
from skimage.data import binary_blobs, chelsea
from skimage.exposure import rescale_intensity
from skimage.util import random_noise
from skimage.measure import compare_psnr, shannon_entropy, compare_mse, compare_nrmse
from skimage.color import rgb2gray, label2rgb
from skimage.restoration import estimate_sigma
from numpy import linalg as LA


class TRND(nn.Module):
    def __init__(self, k=4, T=3, image_channels=1, kernel_size=3):
        super(TRND, self).__init__()
        self.kernel_size = kernel_size
        padding = int((kernel_size-1)/2)
        self.k = k
        self.T = T
        self.diff_layers = nn.ModuleList([nn.Conv2d(
            in_channels=k, out_channels=k, kernel_size=kernel_size, padding=padding, bias=False, groups=k) for i in range(T)])

    def g(self, d_I, mode=1):
        if mode == 1:
            exponent = -torch.abs(d_I)/(1+d_I**2)
            res = torch.exp(exponent)*d1
        elif mode == 2:
            res = torch.reciprocal(1+d_I**2)*d_I
        elif mode == 3:
            res = torch.sqrt(1+d_I**2)*d_I
        return res

    def diffusion(self, x, diff_layer, mode=2):
        n, c, h, w = x.size()
        input_x = torch.zeros(n, self.k, h, w).cuda()+x
        sum_delta = torch.zeros(n, 1, h, w).cuda()
        d = diff_layer(input_x)
        delta_d = self.g(d, mode=mode)
        for i in range(self.k):
            f = delta_d[:, i, :, :]
            f.unsqueeze_(1)
            sum_delta = sum_delta + f
        return sum_delta/self.k

    def forward(self, x):
        in_x = x
        for diff_layer in self.diff_layers:
            in_x = in_x-self.diffusion(in_x, diff_layer)
        y = in_x
        return y


"""
class Conv_TRND(nn.Module):
    def __init__(self, k=4, n_channels=8, image_channels=1, kernel_size=3):
        super(Conv_TRND, self).__init__()
        kernel_size = 3
        padding = 1
        depth = 5
        self.diffs = nn.ModuleList([nn.Conv2d(in_channels=image_channels, out_channels=k,
                            kernel_size=kernel_size, padding=padding, bias=False) for i in range(depth)])
                            
    def g(self, d_I, mode=1):
        if mode == 1:
            exponent = -torch.abs(d_I)/(1+d_I**2)
            res = torch.exp(exponent)
        elif mode == 2:
            res = torch.reciprocal(1+d_I**2)
        elif mode == 3:
            res = torch.sqrt(1+d_I**2)
        return res

    def forward(self, x):
        n, c, h, w = x.size()
        input_x = x
        for f in self.diffs:
            d = f(input_x)
            delta_d = self.g(d,mode=2)*d
            sum_delta_d = torch.zeros(n, 1, h, w).cuda()
            for k in range(c):
                feature = delta_d[:, k, :, :]
                feature.unsqueeze_(1)
                sum_delta_d = sum_delta_d+feature
            input_x = input_x-sum_delta_d
        y = input_x
        return y
"""

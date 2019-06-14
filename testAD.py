import numpy as np
import matplotlib.pyplot as plt
import skimage
import torch
from scipy.ndimage.filters import convolve
from skimage.data import binary_blobs, chelsea
from skimage.exposure import rescale_intensity
from skimage.util import random_noise
from skimage.measure import compare_psnr, shannon_entropy, compare_mse, compare_nrmse
from skimage.color import rgb2gray, label2rgb
from skimage.restoration import estimate_sigma
from numpy import linalg as LA

class AD_F(nn.Module):
    def __init__(self, k=4, T=3, image_channels=1, kernel_size=3):
        super(TRND, self).__init__()
        self.kernel_size = kernel_size
        padding = int((kernel_size-1)/2)
        self.k = k
        self.T = T
        self.padding = padding
        self.diff_layers = nn.ModuleList([nn.Conv2d(
            in_channels=4, out_channels=k, kernel_size=kernel_size, padding=padding, bias=True) for i in range(T)])

    def g(self, d_I, mode=1):
        if mode == 1:
            exponent = -torch.abs(d_I)/(1+d_I**2)
            res = torch.exp(exponent)*d1
        elif mode == 2:
            res = torch.reciprocal(1+d_I**2)*d_I
        elif mode == 3:
            res = torch.sqrt(1+d_I**2)*d_I
        elif mode == 4:
            res = torch.exp(-1*d_I**2)
        return res

    def squash_tensor(self, x):
        n, c, h, w = x.size()
        x_squashed = torch.zeros(n, 1, h, w).cuda()
        for i in range(self.k):
            f = x[:, i, :, :]
            f.unsqueeze_(1)
            x_squashed = x_squashed+f
        return x_squashed

    def extract_feature(self, x):
        n, c, h, w = x.size()
        features = torch.zeros(n, 4, h, w).cuda()
        conv_filter_1 = torch.Tensor([[[[0, 1, 0]], [[0, 0, 0]], [[0, 0, 0]]]])
        conv_filter_2 = torch.Tensor([[[[0, 0, 0]], [[0, 0, 0]], [[0, 1, 0]]]])
        conv_filter_3 = torch.Tensor([[[[0, 0, 0]], [[1, 0, 0]], [[0, 0, 0]]]])
        conv_filter_4 = torch.Tensor([[[[0, 0, 0]], [[0, 0, 1]], [[0, 0, 0]]]])
        feature_filters = [conv_filter_1,
                           conv_filter_2, conv_filter_3, conv_filter_4]
        for i, filter in enumerate(feature_filters):
            filter.transpose_(2, 1)
            f = torch.nn.functional.conv2d(x, filter.cuda(), padding=1)
            t = f.squeeze(1)
            features[:, i, :, :] = t
        return features

    def forward(self, x):
        in_x = x
        for diff_layer in self.diff_layers:
            feature_x = self.extract_feature(in_x)
            d1 = diff_layer(feature_x)
            features = self.g(d1, mode=4)*d1
            in_x = in_x - self.squash_tensor(features)/self.k
        y = in_x
        return y


if __name__ == "__main__":
    img, _ = random.choice(dataset_val)
    total_psnr = 0.0
    i = 0
    for img, _ in dataset_val:
        i = i+1

        img.unsqueeze_(-1)
        img.transpose_(2, 0)
        img.transpose_(3, 1)
        noise = torch.randn(img.shape)*noise_level
        img_n = torch.add(img, noise)
        img_n = Variable(img_n).cuda()
        denoised = autoencoder(img_n)
        img_np = img.squeeze().numpy()
        img_n_np = img_n.squeeze().cpu().detach().numpy()
        denoised_np = denoised.squeeze().cpu().detach().numpy()
        total_psnr + = compare_psnr(img_np, denoised_np)
    print(total_psnr/i)

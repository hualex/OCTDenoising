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
        super(AD_F, self).__init__()
        self.kernel_size = kernel_size
        padding = int((kernel_size-1)/2)
        self.k = k
        self.T = T
        self.padding = padding
        self.diff_layers = nn.ModuleList([nn.Conv2d(
            in_channels=8, out_channels=k, kernel_size=kernel_size, padding=padding, bias=True) for i in range(T)])
        self.prelu_layers = nn.ModuleList([nn.PReLU() for i in range(T)])

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
        conv_filter_1 = torch.Tensor([[[[0, 1, 0]], [[0, 0, 0]], [[0, 0, 0]]]])
        conv_filter_2 = torch.Tensor([[[[0, 0, 0]], [[0, 0, 0]], [[0, 1, 0]]]])
        conv_filter_3 = torch.Tensor([[[[0, 0, 0]], [[1, 0, 0]], [[0, 0, 0]]]])
        conv_filter_4 = torch.Tensor([[[[0, 0, 0]], [[0, 0, 1]], [[0, 0, 0]]]])
        conv_filter_5 = torch.Tensor([[[[1, 0, 0]], [[0, 0, 0]], [[0, 0, 0]]]])
        conv_filter_6 = torch.Tensor([[[[0, 0, 1]], [[0, 0, 0]], [[0, 0, 0]]]])
        conv_filter_7 = torch.Tensor([[[[0, 0, 0]], [[0, 0, 0]], [[1, 0, 0]]]])
        conv_filter_8 = torch.Tensor([[[[0, 0, 0]], [[0, 0, 0]], [[0, 0, 1]]]])
        feature_filters = [conv_filter_1,
                           conv_filter_2, conv_filter_3, conv_filter_4, conv_filter_5, conv_filter_6, conv_filter_7, conv_filter_8]
        features = torch.zeros(n, len(feature_filters), h, w).cuda()
        for i, filter in enumerate(feature_filters):
            filter.transpose_(2, 1)
            f = torch.nn.functional.conv2d(x, filter.cuda(), padding=1)
            t = f.squeeze(1)
            features[:, i, :, :] = t
        return features

    def forward(self, x):
        in_x = x
        for layer_pipeline in self.diff_layers:
            feature_x = self.extract_feature(in_x)
            diff_layer = layer_pipeline[0]
            relu_layer = layer_pipeline[1]
            d1 = diff_layer(feature_x)
            features = relu_layer(d1)*d1
            in_x = in_x - self.squash_tensor(features)/self.k
        y = in_x
        return y


class TRND(nn.Module):
    def __init__(self, k=4, T=3, image_channels=1, kernel_size=3, g_mode=4):
        super(TRND, self).__init__()
        self.kernel_size = kernel_size
        padding = int((kernel_size-1)/2)
        self.k = k
        self.T = T
        self.g_mode = 4
        self.diff_layers = nn.ModuleList([nn.Conv2d(
            in_channels=1, out_channels=k, kernel_size=kernel_size, padding=padding, bias=True) for i in range(T)])
        self.prelu_layers = nn.ModuleList([nn.PReLU() for i in range(T)])

    def g(self, d_I, mode=1):
        if mode == 1:
            exponent = -torch.abs(d_I)/(1+d_I**2)
            res = torch.exp(exponent)
        elif mode == 2:
            res = torch.reciprocal(1+d_I**2)
        elif mode == 3:
            res = torch.sqrt(1+d_I**2)
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

    def forward(self, x):
        in_x = x
        for i in range(self.T):
            diff_layer = self.diff_layers[i]
            prelu_layer = self.prelu_layers[i]
            d_feature = diff_layer(in_x)
            features = prelu_layer(d_feature)*d_feature
            in_x = in_x - self.squash_tensor(features)/self.k
        y = in_x
        return y


if __name__ == "__main__":
    img, _ = random.choice(dataset_val)
    total_psnr = 0.0
    torch.randn()
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
        total_psnr += compare_psnr(img_np, denoised_np)
    print(total_psnr/i)

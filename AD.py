import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as im
import skimage
import torch.nn as nn

from scipy.ndimage.filters import convolve
from skimage.data import binary_blobs, chelsea
from skimage.exposure import rescale_intensity
from skimage.util import random_noise
from skimage.measure import compare_psnr, shannon_entropy, compare_mse, compare_nrmse
from skimage.color import rgb2gray, label2rgb
from skimage.restoration import estimate_sigma


class TD_AD():
    def __init__(self, img, T=10):
        self.img = img
        self.T = T

    def imgradient(self, I):
        a = (2-np.sqrt(2))/4
        b = (np.sqrt(2)-1)/2
        H_x = np.array([[-a, 0, a], [-b, 0, b], [-a, 0, a]])
        H_y = np.array([[-a, -b, -a], [0, 0, 0], [a, b, a]])
        I_x = skimage.filters.gaussian(convolve(I, H_x), sigma=0)
        I_y = skimage.filters.gaussian(convolve(I, H_y), sigma=0)
        #print('graidnet over')
        #print(I_x.shape, I_y.shape)
        return I_x, I_y

    def imhessian(self, I):
        H_xx = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
        H_yy = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])
        H_xy = np.array([[0.25, 0, -0.25], [0, 0, 0], [-0.25, 0, 0.25]])
        I_xx = skimage.filters.gaussian(convolve(I, H_xx), sigma=0)
        I_yy = skimage.filters.gaussian(convolve(I, H_yy), sigma=0)
        I_xy = skimage.filters.gaussian(convolve(I, H_xy), sigma=0)
        #print('hessina over')
        return I_xx, I_yy, I_xy

    def structur_matrix(self, I_x, I_y):
        G_0 = I_x**2
        G_1 = I_x*I_y
        G_2 = I_y**2
        #print('s_matrix over')
        return G_0, G_1, G_2

    def calculate_eigs(self, g):
        w, v = LA.eig(g)
        lambda1 = w[0]
        e1 = np.array([row[0] for row in v])
        lambda2 = w[1]
        e2 = np.array([row[1] for row in v])
        return lambda1, lambda2, e1, e2

    def A_matrix(self, lambda_1, e1, lambda_2, e2, a1=0.5, a2=0.9):
        c1 = 1/np.power(1+lambda_1+lambda_2, a1)
        c2 = 1/np.power(1+lambda_1+lambda_2, a2)
        x1 = e1[0]
        y1 = e1[1]
        a0 = c1*y1**2+c2*x1**2
        a1 = (c2-c1)*x1*y1
        a2 = c1*x1**2+c2*y1**2
        return a0, a1, a2

    def update_beta_k(self, G0, G1, G2, I_xx, I_yy, I_xy):
        m, n = G0.shape
        A0 = np.zeros((m, n))
        A1 = np.zeros((m, n))
        A2 = np.zeros((m, n))
        for i in range(m):
            for j in range(n):
                g_0 = G0[i, j]
                g_1 = G1[i, j]
                g_2 = G2[i, j]
                G_u = np.array([[g_0, g_1], [g_1, g_2]])
                l1, l2, e1, e2 = self.calculate_eigs(G_u)
                a0, a1, a2 = self.A_matrix(l1, e1, l2, e2)
                A0[i, j] = a0
                A1[i, j] = a1
                A2[i, j] = a2
        beta_k = A0*I_xx+2*A1*I_xy+A2*I_yy
        return beta_k

    def update_alpha(self, b, d_t=1):
        return d_t/np.max(np.abs(b))

    def update_image(self, I, alpha, beta_k):
        delta_I = alpha*beta_k
        return I+delta_I

    def apply_diffusion(self):
        I = self.img
        T = self.T
        for _ in range(T):
            I_x, I_y = self.imgradient(I)
            I_xx, I_yy, I_xy = self.imhessian(I)
            G0, G1, G2 = self.structur_matrix(I_x, I_y)
            beta_k = self.update_beta_k(G0, G1, G2, I_xx, I_yy, I_xy)
            alpha = self.update_alpha(beta_k)
            I = self.update_image(I, alpha, beta_k)
        return I

class Deep_AD_F(nn.Module):
    def __init__(self, k=4, T=3, image_channels=1, kernel_size=3):
        super(Deep_AD_F, self).__init__()
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

class PM_AD():
    def __init__(self, img, T):
        self.image = img
        self.T = T

    def g(self, gradient, mode=1, k=60):
        if mode == 4:
            dk_term = np.abs(gradient)/(2.*k)
            dk_term[dk_term > 1] = 1
        else:
            dk_term = np.power(np.abs(gradient)/k, 2)
        result = np.zeros_like(gradient)
        if mode == 1:
            result = np.exp(-1*dk_term)
        elif mode == 2:
            result = 1/(1+dk_term)
        elif mode == 3:
            result = 1/np.sqrt(1+dk_term)
        elif mode == 4:
            result = np.power(1-np.power(dk_term, 2), 2)
        return result

    def dmaps(self, data):
        d0_filter = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
        d0 = convolve(data, d0_filter, mode='reflect')
        d1_filter = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
        d1 = convolve(data, d1_filter, mode='reflect')
        d2_filter = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
        d2 = convolve(data, d2_filter, mode='reflect')
        d3_filter = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
        d3 = convolve(data, d3_filter, mode='reflect')
        return (d0, d1, d2, d3)

    def update_image(self, I, dmap, alpha=0.2, mode=1, k=50):
        """
        Principles_Of_Digital_ImageProces_Advanced
        DOI 10.1007/978-1-84882-919-0
        """
        delta_I = np.zeros_like(I, dtype=float)
        for d in dmap:
            est_k = self.estimate_k(d)
            # print(est_k)
            delta_I += alpha*self.g(d, mode=mode, k=0.2)*d
        return I+delta_I

    def estimate_k(self, dmap):
        k = np.median(np.abs(dmap-np.abs(np.median(dmap))))
        # print(k)
        return k

    def apply_diffusion(self):
        I = self.image
        T = self.T
        for _ in range(T):
            dmaps = self.dmaps(I)
            I = self.update_image(I, dmaps)
        return I

class Deep_AD(nn.Module):
    def __init__(self, k=4, T=3, image_channels=1, kernel_size=3):
        super(Deep_AD, self).__init__()
        self.kernel_size = kernel_size
        padding = int((kernel_size-1)/2)
        self.k = k
        self.T = T
        self.diff_layers = nn.ModuleList([nn.Conv2d(
            in_channels=1, out_channels=k, kernel_size=kernel_size, padding=padding, bias=True) for i in range(T)])

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
        for diff_layer in self.diff_layers:
            d1 = diff_layer(in_x)
            features = self.g(d1, mode=1)*d1
            in_x = in_x - self.squash_tensor(features)/self.k
        y = in_x
        return y

if __name__ == "__main__":
    from PIL import Image
    data_orig = skimage.img_as_float(binary_blobs(length=128, seed=1))
    sigma = 0.2
    data = random_noise(data_orig, var=sigma**2)
    I = data

    diffusion_denoising = PM_AD(I, 100)
    I_d = diffusion_denoising.apply_diffusion()
    #print(compare_mse(data_orig, I), compare_mse(data_orig, I_d))
    #print(compare_ssim(data_orig, I), compare_ssim(data_orig, I_d))
    im.imsave('clean_I.jpeg', data_orig)
    im.imsave('noisy_I.jpeg', data)
    im.imsave('denoised_I.jpeg', I_d)

    plt.figure(figsize=(20, 20))
    plt.subplot(131)
    plt.imshow(data_orig, cmap='gray')
    plt.subplot(132)
    plt.imshow(data, cmap='gray')
    plt.subplot(133)
    plt.imshow(I_d, cmap='gray')
    plt.show()

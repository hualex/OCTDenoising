
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

"""
def g(gradient, mode=1, k=60):
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


def dmaps(data):
    d0_filter = np.array([[0, 0, 0], [0, -1, 1], [0, 0, 0]])
    d0 = convolve(data, d0_filter, mode='reflect')
    d1_filter = np.array([[0, 0, 0], [0, -1, 0], [0, 1, 0]])
    d1 = convolve(data, d1_filter, mode='reflect')
    d2_filter = np.array([[0, 0, 0], [1, -1, 0], [0, 0, 0]])
    d2 = convolve(data, d2_filter, mode='reflect')
    d3_filter = np.array([[0, 1, 0], [0, -1, 0], [0, 0, 0]])
    d3 = convolve(data, d3_filter, mode='reflect')
    return (d0, d1, d2, d3)


def update_image(I, dmap, alpha=0.2, mode=1, k=50):
    """

# Principles_Of_Digital_ImageProces_Advanced
# DOI 10.1007/978-1-84882-919-0

"""
    delta_I = np.zeros_like(I, dtype=float)
    for d in dmap:
        est_k = estimate_k(d)
        delta_I += alpha*g(d, mode=mode, k=est_k)*d
    return I+delta_I


def estimate_k(dmap):
    k = np.median(np.abs(dmap-np.abs(np.median(dmap))))
    # print(k)
    return k


data_orig = skimage.img_as_float(binary_blobs(length=128, seed=1))
# data_orig = skimage.img_as_float(rgb2gray(chelsea()[100:250, 50:300]))
# data_orig = skimage.img_as_float(skimage.io.imread('DME.jpeg'))

sigma = 0.2
data = random_noise(data_orig, var=sigma**2)*255
# print(data)

"""
"""
I = data
for T in range(10000):
    if estimate_sigma(I/255.) <= estimate_sigma(data_orig) or compare_psnr(data_orig, I/255.) < compare_psnr(data_orig, data/255.):
        print("t=", T)
        break
    dmap = dmaps(I)
    I = update_image(I, dmap, mode=2, k=50)
"""
"""
# I = skimage.filters.gabor(data_orig, frequency=100)

I = data
alpha = 0.2
for T in range(1000):
    gradient = skimage.filters.sobel(I)
    I = I+alpha*g(gradient)*gradient
plt.figure(figsize=(20, 20))
plt.subplot(131)
plt.imshow(data_orig, cmap='gray')
plt.subplot(132)
plt.imshow(data, cmap='gray')
plt.subplot(133)
plt.imshow(I, cmap='gray')
plt.show()
print(compare_psnr(data_orig, I/255.), compare_psnr(data_orig, data/255.))
print(compare_mse(data_orig, I/255.), compare_mse(data_orig, data/255.))
print(estimate_sigma(data/255.), estimate_sigma(data_orig), estimate_sigma(I/255.))

"""


class TRND(nn.Module):
    def __init__(self, k=10, n_channels=8, image_channels=1, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        self.difflayer = nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
                                   kernel_size=kernel_size, padding=padding, bias=True)
        self.difflayer = nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
                                   kernel_size=kernel_size, padding=padding, bias=True)

    def forward(self, x):
        feature_map = self.difflayer(x)
        print(feature_map.size)

        return feature_map


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        kernel_size = 3
        padding = 1
        pipe_line = []
        pipe_line.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
                                   kernel_size=kernel_size, padding=padding, bias=True))
        pipe_line.append(nn.ReLU(inplace=True))
        pipe_line.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
                                   kernel_size=kernel_size, padding=padding, bias=True))
        self.dncnn1 = nn.Sequential(*pipe_line)

        self.hidden_size = hidden_size

        self.i2h = nn.Sequential(*pipe_line)
        #self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        #output = self.i2o(combined)
        #output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)


class TD_AD():
    def __init__(self, img, T=10):
        self.img = img
        self.T = T

    def imgradient(I):
        a = (2-np.sqrt(2))/4
        b = (np.sqrt(2)-1)/2
        H_x = np.array([[-a, 0, a], [-b, 0, b], [-a, 0, a]])
        H_y = np.array([[-a, -b, -a], [0, 0, 0], [a, b, a]])
        I_x = skimage.filters.gaussian(convolve(I, H_x), sigma=0)
        I_y = skimage.filters.gaussian(convolve(I, H_y), sigma=0)
        print('graidnet over')
        print(I_x.shape, I_y.shape)
        return I_x, I_y

    def imhessian(I):
        H_xx = np.array([[0, 0, 0], [1, -2, 1], [0, 0, 0]])
        H_yy = np.array([[0, 1, 0], [0, -2, 0], [0, 1, 0]])
        H_xy = np.array([[0.25, 0, -0.25], [0, 0, 0], [-0.25, 0, 0.25]])
        I_xx = skimage.filters.gaussian(convolve(I, H_xx), sigma=0)
        I_yy = skimage.filters.gaussian(convolve(I, H_yy), sigma=0)
        I_xy = skimage.filters.gaussian(convolve(I, H_xy), sigma=0)
        print('hessina over')
        return I_xx, I_yy, I_xy

    def structur_matrix(I_x, I_y):
        G_0 = I_x**2
        G_1 = I_x*I_y
        G_2 = I_y**2
        print('s_matrix over')
        return G_0, G_1, G_2

    def calculate_eigs(g):
        w, v = LA.eig(g)
        lambda1 = w[0]
        e1 = np.array([row[0] for row in v])
        lambda2 = w[1]
        e2 = np.array([row[1] for row in v])
        return lambda1, lambda2, e1, e2

    def A_matrix(lambda_1, e1, lambda_2, e2, a1=0.5, a2=0.9):
        c1 = 1/np.power(1+lambda_1+lambda_2, a1)
        c2 = 1/np.power(1+lambda_1+lambda_2, a2)
        x1 = e1[0]
        y1 = e1[1]
        a0 = c1*y1**2+c2*x1**2
        a1 = (c2-c1)*x1*y1
        a2 = c1*x1**2+c2*y1**2
        return a0, a1, a2

    def update_beta_k(G0, G1, G2, I_xx, I_yy, I_xy):
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
                l1, l2, e1, e2 = calculate_eigs(G_u)
                a0, a1, a2 = A_matrix(l1, e1, l2, e2)
                A0[i, j] = a0
                A1[i, j] = a1
                A2[i, j] = a2
        beta_k = A0*I_xx+2*A1*I_xy+A2*I_yy
        return beta_k

    def update_alpha(b, d_t=1):
        return d_t/np.max(np.abs(b))

    def update_image(I, alpha, beta_k):
        delta_I = alpha*beta_k
        return I+delta_I


if __name__ == "__main__":
    data_orig = skimage.img_as_float(binary_blobs(length=128, seed=1))
    sigma = 0.2
    data = random_noise(data_orig, var=sigma**2)*255
    I = data
    for T in range(100):
        print(T)
        I_x, I_y = imgradient(I)
        I_xx, I_yy, I_xy = imhessian(I)
        G0, G1, G2 = structur_matrix(I_x, I_y)
        beta_k = update_beta_k(G0, G1, G2, I_xx, I_yy, I_xy)
        alpha = update_alpha(beta)
        I = update_image(I, alpha, beta_k)
        plt.figure(figsize=(20, 20))
        plt.subplot(131)
        plt.imshow(data_orig, cmap='gray')
        plt.subplot(132)
        plt.imshow(data, cmap='gray')
        plt.subplot(133)
        plt.imshow(I, cmap='gray')
        plt.show()

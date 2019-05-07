
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

    Principles_Of_Digital_ImageProces_Advanced
    DOI 10.1007/978-1-84882-919-0


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
#data_orig = skimage.img_as_float(rgb2gray(chelsea()[100:250, 50:300]))
#data_orig = skimage.img_as_float(skimage.io.imread('DME.jpeg'))

sigma = 0.2
data = random_noise(data_orig, var=sigma**2)*255
# print(data)


I = data
for T in range(10000):
    if estimate_sigma(I/255.) <= estimate_sigma(data_orig) or compare_psnr(data_orig, I/255.) < compare_psnr(data_orig, data/255.):
        print("t=", T)
        break
    dmap = dmaps(I)
    I = update_image(I, dmap, mode=2, k=50)
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
Todo:
1. Stopping time ?
2. other g function ?
3. optimum setting for alpha,k through machine learning ?
"""

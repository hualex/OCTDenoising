
import numpy as np
import matplotlib.pyplot as plt
import skimage

from scipy.ndimage.filters import convolve

from skimage.data import binary_blobs
from skimage.exposure import rescale_intensity
from skimage.util import random_noise


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
    delta_I = np.zeros_like(I, dtype=float)
    for d in dmap:
        delta_I += alpha*g(d, mode=mode, k=k)*d
    return I+delta_I


data = skimage.img_as_float(binary_blobs(length=128, seed=1))
sigma = 0.35
data = random_noise(data, var=sigma**2)*255
# print(data)


I = data
for T in range(100):
    dmap = dmaps(I)
    I = update_image(I, dmap, mode=4, k=60)

plt.subplot(131)
plt.imshow(skimage.img_as_float(binary_blobs(length=128, seed=1)), cmap='gray')
plt.subplot(132)
plt.imshow(I, cmap='gray')
plt.subplot(133)
plt.imshow(data, cmap='gray')
plt.show()

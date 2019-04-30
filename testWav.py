
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import matplotlib.pyplot as plt
import pywt
import heapq
import glob
import os
import time
import numpy as np


from skimage.restoration import denoise_wavelet, cycle_spin
from skimage import data, img_as_float
from skimage.util import random_noise
from skimage.measure import compare_psnr
from skimage import io


def combine_dicts(a, b):
    return {x: a.get(x, 0)+b.get(x, 0) for x in set(a).union(b)}


def scale_dicts_with_factor(d, factor):
    factor = factor/sum(d.values())
    for k in d:
        d[k] = d[k]*factor


def CountFrequency(my_list):

    # Creating an empty dictionary
    freq = {}
    for item in my_list:
        if (item in freq):
            freq[item] += 1
        else:
            freq[item] = 1
        """
    for key, value in freq.items():
        print("% d : % d" % (key, value))
        """
    return freq


"""
x = {'a':1, 'b': 2}
y = {'b':10, 'c': 11}
print(merge_two_dicts(x,y))

"""


def top_ten_wrt_psnr(img, wave_list, sigma=0.1, n_largest=3):
    original = img_as_float(img)
    noisy = random_noise(original, var=sigma**2)

    # Repeat denosing with different amounts of cycle spinning.  e.g.
    # max_shift = 0 -> no cycle spinning
    # max_shift = 1 -> shifts of (0, 1) along each axis
    # max_shift = 3 -> shifts of (0, 1, 2, 3) along each axis
    # etc...
    psnr_data = []

    for wave in wave_list:
        denoise_kwargs = dict(multichannel=False,
                              convert2ycbcr=False, wavelet=wave)
        all_psnr = []
        max_shifts = [0, 1, 3, 5]
        for n, s in enumerate(max_shifts):
            im_bayescs = cycle_spin(noisy, func=denoise_wavelet, max_shifts=s,
                                    func_kw=denoise_kwargs, multichannel=True)
            psnr = compare_psnr(original, im_bayescs)
            all_psnr.append(psnr)
        psnr_data.append(all_psnr)

    p = np.array(psnr_data)
    maxP = np.max(p, axis=1)

    index_wave_p = heapq.nlargest(
        n_largest, range(len(maxP)), key=maxP.__getitem__)
    name_wave_p = [wave_list[i] for i in index_wave_p]
    print(name_wave_p)
    return name_wave_p


def denoising_with_wave(wave, img, sigma=0.1):
    original = img_as_float(img)
    noisy = random_noise(original, var=sigma**2)
    denoise_kwargs = dict(multichannel=False,
                          convert2ycbcr=False, wavelet=wave)
    all_psnr = []
    max_shifts = [0, 1, 3, 5]
    for n, s in enumerate(max_shifts):
        im_bayescs = cycle_spin(noisy, func=denoise_wavelet,
                                max_shifts=s, func_kw=denoise_kwargs, multichannel=True)
        psnr = compare_psnr(original, im_bayescs)
        all_psnr.append(psnr)

    return (wave, all_psnr)


if __name__ == "__main__":
    # os.chdir(r"C:\Users\Hualex\Google Drive\OCT2017\test\CNV")
    # pathname='./gdrive/My Drive/OCT2017/test/DME/*.jpeg'
    #files = glob.glob(pathname)
    #print('flie number:', len(files))
    name_dict = {}
    name_dict_second = {}
    name_dict_third = {}
    i = 0
    pathname = './gdrive/My Drive/OCT2017/test/'
    T = transforms.Compose([transforms.Grayscale(),
                            transforms.CenterCrop(450),
                            transforms.ToTensor()])
    dataset = ImageFolder(pathname, transform=T)
    dataloder = DataLoader(dataset, batch_size=10, shuffle=True, num_workers=2)
    wave_list = []
    dwave_families = ['haar', 'db', 'sym', 'coif', 'bior', 'rbio']
    for wave in dwave_families:
        wave_list = wave_list+pywt.wavelist(wave)
    i = 0
    for data in dataloder:
        s = time.time()
        if i == 3:
            break
        imgs = data[0].numpy()
        name_list = []
        for img in imgs:
            image = np.squeeze(img)
            # plt.imshow(image)
            # plt.show()
            # print(image)
            p_list = top_ten_wrt_psnr(image, wave_list)
            name_list.append(p_list)
        name_dict = combine_dicts(CountFrequency(
            [row[0] for row in name_list]), name_dict)
        name_dict_second = combine_dicts(CountFrequency(
            [row[1] for row in name_list]), name_dict_second)
        name_dict_third = combine_dicts(CountFrequency(
            [row[2] for row in name_list]), name_dict_third)
        i = i+1
        print("Processin Time for 10 Images:", time.time()-s)
    scale_dicts_with_factor(name_dict_second, 0.3)
    scale_dicts_with_factor(name_dict, 0.5)
    scale_dicts_with_factor(name_dict_third, 0.2)
    tempd = combine_dicts(name_dict, name_dict_second)
    tempd = combine_dicts(tempd, name_dict_third)

    f1 = plt.figure()
    plt.bar(tempd.keys(), tempd.values(), color='g')
    plt.show()

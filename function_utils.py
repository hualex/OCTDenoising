import torch
import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
import cv2
import glob
import torch.utils.data as udata
from PIL import Image
from torchvision import datasets
from torchvision import transforms
from scipy import ndimage
from random import sample as sp, shuffle as sf


class Preprocessing(object):

    def __call__(self, sample):

        img = self.rgb2gray(np.asarray(sample))
        img_mean_map, img_dev_map = self.get_local_map(img)
        result = (img-img_mean_map)/img_dev_map
        return Image.fromarray(np.uint8(img))

    def rgb2gray(self, rgb):
        return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])

    def get_local_map(self, img, kernel_size=3):

        img_local_mean_map = ndimage.uniform_filter(
            img, size=kernel_size, mode="reflect")
        img_sqr_map = ndimage.uniform_filter(
            img**2, size=kernel_size, mode="reflect")
        img_local_var_map = img_sqr_map-img_local_mean_map**2
        img_local_std_map = np.sqrt(img_local_var_map)
        return img_local_mean_map, img_local_std_map


def show_vae_img(orig, noisy, denoised, image_shape):
    """

    """
    orig = np.reshape(orig, image_shape)
    noisy = np.reshape(noisy, image_shape)
    denoised = np.reshape(denoised, image_shape)
    orig = (orig - orig.min()) / (orig.max() - orig.min())
    noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())
    denoised = (denoised - denoised.min()) / (denoised.max() - denoised.min())
    fig = plt.figure()

    fig.add_subplot(1, 3, 1, title='Original')
    plt.imshow(orig, cmap='gray')

    fig.add_subplot(1, 3, 2, title='Noisy')
    plt.imshow(noisy, cmap='gray')

    fig.add_subplot(1, 3, 3, title='Denoised')
    plt.imshow(denoised, cmap='gray')


def show_img(orig, noisy, denoised):
    """
    show original image, noisy image and denoised image for comparison purposes

    """
    fig = plt.figure()
    # for 3 channel image
    orig = orig.swapaxes(0, 1).swapaxes(1, 2)
    noisy = noisy.swapaxes(0, 1).swapaxes(1, 2)
    denoised = denoised.swapaxes(0, 1).swapaxes(1, 2)

    # for 1 channel image
    #orig = orig.squeeze()
    #noisy = noisy.squeeze()
    #denoised = denoised.squeeze()

    # Normalize for display purpose
    orig = (orig - orig.min()) / (orig.max() - orig.min())
    noisy = (noisy - noisy.min()) / (noisy.max() - noisy.min())
    denoised = (denoised - denoised.min()) / (denoised.max() - denoised.min())

    fig.add_subplot(1, 3, 1, title='Original')
    plt.imshow(orig, cmap='gray')

    fig.add_subplot(1, 3, 2, title='Noisy')
    plt.imshow(noisy, cmap='gray')

    fig.add_subplot(1, 3, 3, title='Denoised')
    plt.imshow(denoised, cmap='gray')

    #fig.subplots_adjust(wspace = 1)
    # plt.show()


def get_dataset(image_path, crop_size, train_size, test_size, batch_size=10, number_output_channels=3):
    """
    get image patch dataset through centercrop 

    """
    transform = transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    total_dataset = datasets.ImageFolder(image_path, transform=transform)
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(total_dataset,
                                                                             [train_size, test_size, len(total_dataset)-train_size-test_size])
    dataset_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=1)
    dataset_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                      shuffle=True, num_workers=1)
    dataset_valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=1)
    return dataset_train_loader, dataset_test_loader, dataset_valid_loader


def get_dataset_vae(image_path, crop_size, train_size, test_size, batch_size=10, number_output_channels=3):
    """
    get image patch dataset through centercrop 

    """
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Lambda(lambda x:x.view(-1))])
    total_dataset = datasets.ImageFolder(image_path, transform=transform)
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(total_dataset,
                                                                             [train_size, test_size, len(total_dataset)-train_size-test_size])
    dataset_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=1)
    dataset_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                                      shuffle=True, num_workers=1)
    dataset_valid_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                                       shuffle=True, num_workers=1)
    return dataset_train_loader, dataset_test_loader, dataset_valid_loader
# To test
#o = oct_train_dataset[80]
#a,b,c = o[0].size()
# print(a,b,c)
# show_img(o[0].numpy(),o[0].numpy(),o[0].numpy())


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    print(patch.shape)
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def generate_h5_file(pathname, file_format='jpeg', file_number=500, train=True):
    dir_list = ['CNV', 'DME', 'DRUSEN', 'NORMAL']
    files = []
    for dir in dir_list:
        tempfiles = glob.glob(os.path.join(pathname, dir, '*.'+file_format))
        for tempfile in tempfiles:
            files.append((tempfile, dir))
    if train:
        h5f = h5py.File('train.h5', 'w')
    else:
        h5f = h5py.File('val.h5', 'w')
    print("total file number: ", len(files))
    val_num = 0
    sampled_files = sp(files, file_number)
    for i in range(len(sampled_files)):
        file_with_label = sampled_files[i]
        img = cv2.imread(file_with_label[0])
        print(img.shape)
        label = dir_list.index(file_with_label[1])
        g = h5f.create_group(str(val_num))
        g.create_dataset('data', data=img)
        g.create_dataset('label', data=label, dtype=np.intp)
        val_num += 1
    h5f.close()


class Dataset(udata.Dataset):
    def __init__(self, train=True, transform=None, denoising=True):
        super(Dataset, self).__init__()
        self.transform = transform
        self.train = train
        self.denoising = denoising

        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())
        sf(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        data_key = 'data'
        label_key = 'label'
        data = np.array(h5f[key][data_key])
        data = Image.fromarray(data)
        if self.transform:
            data = self.transform(data)
        label = np.array(h5f[key][label_key])
        h5f.close()
        if self.denoising:
            return data
        else:
            return data, label

    def normalize(self, data):
        return data/255.

    def swapaxies(self, img):
        img = img.swapaxes(0, 2)
        return img

    def Im2Patch(self, img, win=320, stride=1):
        patch_num = 0
        w, h, c = img.shape
        img_patches = []
        for x in range(0, h-win, stride):
            for y in range(0, w-win, stride):
                window = img[x:x + win, y:y + win, :]
                img_patches.append(window)
                patch_num += 1
        return img_patches, patch_num

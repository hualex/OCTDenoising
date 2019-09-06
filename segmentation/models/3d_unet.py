import h5py
import torch
import numpy as np
from torch.utils.data import Dataset, random_split as r_split
from unet_parts import convBlock, downBlock, upBlock, bottomBlock, outBlock
from torch import nn
from torchvision import transforms


class H5Dataset(Dataset):
    def __init__(self, H5FilePath, transform=None):
        self.h5f = h5py.File(H5FilePath, 'r')
        self.voxel_list = list(self.h5f.keys())
        self.transform = transform

    def __len__(self):
        return len(self.voxel_list)

    def __getitem__(self, idx):
        idx_key = self.voxel_list[idx]
        data = np.swapaxes(self.h5f[idx_key], 0, 1)
        if self.transform:
            data = self.transform(data)
        return data


class UNet_3D(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet_3D, self).__init__()
        self.conv_in = nn.Sequential(
            convBlock(1, 3),
            convBlock(3, 8),
            convBlock(8, 16)
        )
        self.down_1 = downBlock(16, 32)
        self.down_2 = downBlock(32, 64)
        self.up_1 = upBlock(96, 32)
        self.up_2 = upBlock(192, 64)
        self.bottom = bottomBlock(64, 128)
        self.conv_out = outBlock(48, 16)

    def forward(self, x):
        x_in = self.conv_in(x)
        x_donw_1 = self.down_1(x_in)
        x_donw_2 = self.down_2(x_donw_1)
        x_bottom = self.bottom(x_donw_2)
        x_up_2 = self.up_2(x_donw_2, x_bottom)
        x_up_1 = self.up_1(x_donw_1, x_up_2)
        y = self.conv_out(x_in, x_up_1)
        return y


if __name__ == "__main__":
    filePath = '/c/Users/Hualex/Documents/MRI/kaggle_3m/MRI_Image_Voxel_c1.h5'
    dataset_voxel = H5Dataset(filePath)
    for data in dataset_voxel:
        print(data.shape)

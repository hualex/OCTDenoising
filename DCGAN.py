from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML


class ResidueBlock(nn.Module):
    def __init__(self, n_channels=6, kernel_size=3, padding=1):
        super(ResidueBlock, self).__init__()
        before_sum_layers = []
        after_sum_layers = []
        for _ in range(2):
            before_sum_layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                                               kernel_size=kernel_size, padding=padding, bias=False))
            before_sum_layers.append(nn.BatchNorm2d(
                n_channels, eps=0.0001, momentum=0.95))
            before_sum_layers.append(nn.ReLU(inplace=True))
        after_sum_layers.append(nn.BatchNorm2d(
            n_channels, eps=0.0001, momentum=0.95))
        after_sum_layers.append(nn.ReLU(inplace=True))
        after_sum_layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                                          kernel_size=kernel_size, padding=padding, bias=False))
        self.before_sum = nn.Sequential(*before_sum_layers)
        self.after_sum = nn.Sequential(*after_sum_layers)

    def forward(self, x):
        y = self.before_sum(x)
        out = self.after_sum(y+x)
        return out


class DCGAN_Generator(nn.Module):
    def __init__(self, depth=5, n_channels=6, image_channels=1, kernel_size=3, padding=1):
        super(DCGAN_Generator, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels,
                                kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth):
            layers.append(ResidueBlock(n_channels=n_channels,
                                       kernel_size=kernel_size, padding=padding))
        for _ in range(2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels,
                                kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.Sigmoid())

        self.dcgan = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dcgan(x)
        return out


class OCTResNet(nn.Module):
    def __init__(self):
        super(OCTResNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 3,
                                     kernel_size=(3, 3),
                                     stride=(2, 2),
                                     padding=(0, 0), bias=False)
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.minst_resnet = torch.nn.Sequential(self.conv1, self.resnet18)

    def forward(self, x):
        return self.minst_resnet(x)


class DCGAN_Discriminator(nn.Module):
    def __init__(self):
        super(DCGAN_Discriminator, self).__init__()
        self.dis = nn.Sequential(OCTResNet(),
                                 nn.Linear(1000, 1),
                                 nn.Tanh())

    def forward(self, x):
        return self.dis(x)


class DCGAN(nn.Module):
    def __init__(self):
        super(DCGAN, self).__init__()
        self.generator = DCGAN_Generator()
        self.discriminator = DCGAN_Discriminator()

    def generator_forward(self, z):
        img = self.generator(z)
        return img

    def discriminator_forward(self, img):
        pred = self.discriminator(img)
        return pred.view(-1)

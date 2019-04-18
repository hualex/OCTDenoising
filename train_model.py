import torch
import torch.cuda as cuda
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import cv2
import glob
import h5py

from torch.autograd import Variable
from models import DnCNN,VAE
from function_utils import show_img,get_dataset,show_vae_img
from test import prepare_data,Dataset,generate_h5_file,noise_generator
from torch.nn import functional as F
from torchvision.transforms import LinearTransformation
from preprocessing import rgb2gray,get_local_map


#!ls ./gdrive/My\ Drive/OCT2017/train/

def loss_func_DnCNN(output, target):
    mse_image = nn.functional.mse_loss(output,image)
    output_local_mean_map,output_local_dev_map = get_local_map(output)
    target_local_mean_map,target_local_dev_map = get_local_map(target)
    mse_mean_map = nn.functional.mse_loss(output_local_mean_map,target_local_mean_map)
    mse_dev_map = nn.functional.mse_loss(output_local_dev_map,target_local_dev_map)
    return mse_image+mse_mean_map+mse_dev_map




train_pathname = './gdrive/My Drive/OCT2017/train'
test_pathname = './gdrive/My Drive/OCT2017/test'
train_size = 100
valid_size = 10


#dataset_val = Dataset(train=False)
#dataset_train = Dataset()

#dataset_train,dataset_val,_ = get_dataset(train_pathname,crop_size=451,train_size=train_size,test_size=valid_size)
dataset_train,dataset_val,_ = get_dataset(train_pathname,crop_size=451,train_size=train_size,
                                        test_size=valid_size,batch_size=10)

DenoiseModel = DnCNN(depth=10,n_channels=12,kernel_size=3).cuda()
parameters = list(autoencoder.parameters())
noise_level = 0.1
learning_rate =0.001
optimizer = torch.optim.Adam(parameters, lr=learning_rate)

train_loss = []
valid_loss = []
time_epoch = []
train_psnr = []
valid_psnr = []
gpu_size = []
v_gpu_size = []

epoche_number = 10
for i in range(epoche_number):
    
    # Let's train the model
    s = time.time() 
    total_loss = 0.0
    total_iter = 0
    j = 0
    DenoiseModel.train()
    for image in dataset_train:      
        j+=1      
        #print('Batch iter: {} Beging traning GPU Memory allocated: {} MB'.format(j,torch.cuda.memory_allocated() / 1024**2))
        gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
        image = image.resize_(1,image.shape[0],image.shape[1],image.shape[2])        
        noise = torch.randn(image.shape)*noise_level
        image_n = torch.add(image,noise)
        image = Variable(image).cuda()
        image_n = Variable(image_n).cuda()
        #print('Batch iter: {} before training traning GPU Memory allocated: {} MB'.format(j,torch.cuda.memory_allocated() / 1024**2))
        gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
       

        optimizer.zero_grad()
        output = DenoiseModel(image_n)
        
        loss = loss_func(output, image)
        loss.backward()
        #print('Batch iter: {} after training traning GPU Memory allocated: {} MB'.format(j,torch.cuda.memory_allocated() / 1024**2))
        gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
        optimizer.step()

        
        total_iter += 1
        total_loss += loss.item()
    #print('Epoch:{} GPU Memory allocated: {} MB'.format(i,torch.cuda.memory_allocated() / 1024**2))
        
     
        
    # Let's record the validation loss
    
    total_val_loss = 0.0
    total_val_iter = 0
    DenoiseModel.eval()
    for image in dataset_val:
        v_gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
        image = image.resize_(1,image.shape[0],image.shape[1],image.shape[2])
        noise = torch.randn(image.shape)*noise_level
        image_n = torch.add(image,noise)
        image = Variable(image).cuda()
        image_n = Variable(image_n).cuda()
        #print('Eval GPU Memory allocated: {} MB'.format(torch.cuda.memory_allocated() / 1024**2))
        v_gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
        
        output = DenoiseModel(image_n)
        loss = loss_func(output, image)
        
        total_val_iter += 1
        total_val_loss += loss.detach().item()
        v_gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
        
        
    train_loss.append(total_loss / total_iter)
    valid_loss.append(total_val_loss / total_val_iter)
    e = time.time()
    print("Epoche", i+1)
    print('GPU Memory allocated: {} MB'.format(torch.cuda.memory_allocated() / 1024**2))
    print("Time elapsed:",e-s)
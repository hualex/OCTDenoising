from google.colab import drive 
drive.mount('/content/gdrive')

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
from models import DenoisingAutoencoder,DnCNN,VAE
from function_utils import show_img,get_dataset,show_vae_img
from test import prepare_data,Dataset,generate_h5_file,noise_generator
from torch.nn import functional as F
from torchvision.transforms import LinearTransformation

train_pathname = './gdrive/My Drive/OCT2017/train'
test_pathname = './gdrive/My Drive/OCT2017/test'
train_size = 100
valid_size = 10


#dataset_val = Dataset(train=False)
#dataset_train = Dataset()

#dataset_train,dataset_val,_ = get_dataset(train_pathname,crop_size=451,train_size=train_size,test_size=valid_size)
dataset_train,dataset_val,_ = get_dataset(train_pathname,crop_size=451,train_size=train_size,test_size=valid_size,batch_size=10,number_output_channels=1)


#autoencoder = DenoisingAutoencoder().cuda()
#autoencoder.load_state_dict(torch.load("./gdrive/My Drive/PTHFILE/DnCNN_D10_K3_n12.pth"))
autoencoder = VAE(image_size=451*451).cuda()
noise_level = 0.1
learning_rate =0.001
#autoencoder= DnCNN(depth=10,n_channels=12,kernel_size=3).cuda()
parameters = list(autoencoder.parameters())
#loss_func = autoencoder.loss_function()
optimizer = torch.optim.Adam(parameters, lr=learning_rate)
#del autoencoder
torch.cuda.empty_cache()



train_loss = []
valid_loss = []
time_epoch = []
gpu_size = []
v_gpu_size = []

epoche_number = 30
for i in range(epoche_number):
    
    # Let's train the model
    s = time.time() 
    total_loss = 0.0
    total_iter = 0
    j = 0
    autoencoder.train()
    for image,_ in dataset_train:      
        j+=1      
        #print('Batch iter: {} Beging traning GPU Memory allocated: {} MB'.format(j,torch.cuda.memory_allocated() / 1024**2))
        gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
    
        noise = torch.randn(image.shape)*noise_level
        
        
        image_n = torch.add(image,noise)
        image = Variable(image).cuda()
        image_n = Variable(image_n).cuda()
        #print('Batch iter: {} before training traning GPU Memory allocated: {} MB'.format(j,torch.cuda.memory_allocated() / 1024**2))
        gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
  
       

        optimizer.zero_grad()
        #output = autoencoder(image_n)
        output,mu,log_var = autoencoder(image_n)
        
        loss = autoencoder.loss_function(output, image,mu,log_var)
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
    autoencoder.eval()
    for image,_ in dataset_val:
        v_gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
        #image = image.resize_(1,image.shape[0],image.shape[1],image.shape[2])
        #noise = torch.randn(1,image.shape[1],image.shape[2],image.shape[3])*noise_level
        #noise = torch.randn(image.shape[0],image.shape[1],image.shape[2],image.shape[3])*noise_level
        noise = torch.randn(image.shape)*noise_level
        image_n = torch.add(image,noise)
        image = Variable(image).cuda()
        image_n = Variable(image_n).cuda()
        #print('Eval GPU Memory allocated: {} MB'.format(torch.cuda.memory_allocated() / 1024**2))
        v_gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
        
        #output = autoencoder(image_n)
        output,mu,log_var = autoencoder(image_n)
        loss = autoencoder.loss_function(output, image,mu,log_var)
        #loss = loss_func(output, image)
        
        total_val_iter += 1
        total_val_loss += loss.detach().item()
        v_gpu_size.append(torch.cuda.memory_allocated() / 1024**2)
        
    
    train_loss.append(total_loss / total_iter)
    valid_loss.append(total_val_loss / total_val_iter)
    e = time.time()
    print("Iteration ", i+1)
    print('GPU Memory allocated: {} MB'.format(torch.cuda.memory_allocated() / 1024**2))
    print("Time elapsed:",e-s)

fig = plt.figure(figsize=(10, 7))
plt.plot(train_loss, label='Train loss')
plt.plot(valid_loss, label='Validation loss')
plt.legend()
plt.show()
train_loss
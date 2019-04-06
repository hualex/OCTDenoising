import torch
import matplotlib.pyplot as plt
import numpy as np
from torchvision import datasets
from torchvision import transforms


def show_img(orig, noisy, denoised):
    fig=plt.figure()
    ### for 3 channel image
    orig = orig.swapaxes(0, 1).swapaxes(1, 2)
    noisy = noisy.swapaxes(0, 1).swapaxes(1, 2)
    denoised = denoised.swapaxes(0, 1).swapaxes(1, 2)
    
    ### for 1 channel image
    #orig = orig.squeeze()
    #noisy = noisy.squeeze()
    #denoised = denoised.squeeze()
    
    
    # Normalize for display purpose
    orig     = (orig - orig.min()) / (orig.max() - orig.min())
    noisy    = (noisy - noisy.min()) / (noisy.max() - noisy.min())
    denoised = (denoised - denoised.min()) / (denoised.max() - denoised.min())

    
    fig.add_subplot(1, 3, 1, title='Original')
    plt.imshow(orig,cmap='gray')
    
    fig.add_subplot(1, 3, 2, title='Noisy')
    plt.imshow(noisy,cmap = 'gray')
    
    fig.add_subplot(1, 3, 3, title='Denoised')
    plt.imshow(denoised,cmap = 'gray')
    
    #fig.subplots_adjust(wspace = 1)
    #plt.show()


def get_dataset(image_path,crop_size,train_size,test_size,batch_size=10):
    transform=transforms.Compose([
        transforms.CenterCrop(crop_size),
        transforms.ToTensor()])
    total_dataset = datasets.ImageFolder(image_path,transform=transform)
    train_dataset, test_dataset, val_dataset = torch.utils.data.random_split(total_dataset,
    [train_size, test_size,len(total_dataset)-train_size-test_size])
    dataset_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, 
    shuffle=True, num_workers=1)
    dataset_test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, 
    shuffle=True, num_workers=1)
    dataset_valid_loader = torch.utils.data.DataLoader(val_dataset,batch_size=batch_size,
    shuffle=True,num_workers=1)
    return dataset_train_loader,dataset_test_loader,dataset_valid_loader


# To test
#o = oct_train_dataset[80]
#a,b,c = o[0].size()
#print(a,b,c)
#show_img(o[0].numpy(),o[0].numpy(),o[0].numpy())

def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    print(patch.shape)
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


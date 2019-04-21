import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata




def Im2Patch(img, win, stride=1):
    patch_num = 0
    w,h,c = img.shape
    img_patches = []
    for x in range(0, h-win , stride):
        for y in range(0, w-win, stride):
            window = img[x:x + win, y:y + win, :]
            img_patches.append(window)
            patch_num+=1
    return img_patches, patch_num

      

def generate_h5_file(pathname,file_format='jpeg',file_number=500,train=True):
    dir_list = ['CNV','DME','DRUSEN','NORMAL']
    files = []
    for dir in dir_list:
        tempfiles = glob.glob(os.path.join(pathname, dir, '*.'+file_format))
        for tempfile in tempfiles:
            files.append((tempfile,dir))
    if train:
        h5f = h5py.File('train.h5', 'w')
    else:
        h5f = h5py.File('val.h5', 'w')
    print("total file number: ",len(files))
    val_num = 0
    sampled_files = random.sample(files,file_number)
    for i in range(len(sampled_files)):
        file_with_label = sampled_files[i]
        img = cv2.imread(file_with_label[0])
        print(img.shape)
        label = dir_list.index(file_with_label[1])
        g = h5f.create_group(str(val_num))                        
        g.create_dataset('data', data=img)
        g.create_dataset('label', data=label,dtype=np.intp)    
        val_num += 1   
    h5f.close()


class Dataset(udata.Dataset):
    def __init__(self, train=True,transform = None,denoising=True):
        super(Dataset, self).__init__()
        self.transform = transform        
        self.train = train
        self.denoising = denoising

        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()
    def __len__(self):        
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('train.h5', 'r')
        else:
            h5f = h5py.File('val.h5', 'r')
        key = self.keys[index]
        data_key ='data'
        label_key ='label'
        data = np.array(h5f[key][data_key])
        data = torch.Tensor(self.swapaxies(self.normalize(data)))
        if self.transform:
            data = self.transform(data)
        label = np.array(h5f[key][label_key])
        h5f.close()
        if self.denoising:
            return data
        else:
            return data,label 
        

    def normalize(self,data):
        return data/255.

    def swapaxies(self,img):
        img = img.swapaxes(0, 2)
        return img

    def Im2Patch(self,img, win=320, stride=1):
        patch_num = 0
        w,h,c = img.shape
        img_patches = []
        for x in range(0, h-win , stride):
            for y in range(0, w-win, stride):
                window = img[x:x + win, y:y + win, :]
                img_patches.append(window)
                patch_num+=1
        return img_patches, patch_num

def save_checkpoint(state, is_best, filename='/output/checkpoint.pth.tar'):
    """Save checkpoint if a new best is achieved"""
    if is_best:
        print ("=> Saving a new best")
        torch.save(state, filename)  # save checkpoint
    else:
        print ("=> Validation Accuracy did not improve")



def resume_checkpoint(resume_weights):

    # cuda = torch.cuda.is_available()
    if cuda:
        checkpoint = torch.load(resume_weights)

    # Load GPU model on CPU
    else:
        checkpoint = torch.load(resume_weights,
                            map_location=lambda storage,
                            loc: storage)
    start_epoch = checkpoint['epoch']
    best_accuracy = checkpoint['best_accuracy']
    model.load_state_dict(checkpoint['state_dict'])
    print("=> loaded checkpoint '{}' (trained for {} epochs)".format(resume_weights, checkpoint['epoch']))







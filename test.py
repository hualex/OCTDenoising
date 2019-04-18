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
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win*win,TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:,i:endw-win+i+1:stride,j:endh-win+j+1:stride]
            Y[:,k,:] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])

def prepare_data(data_path, patch_size, stride, aug_times=1):
    # train
    print('process training data')
    scales = [1, 0.9, 0.8, 0.7]
    files = glob.glob(os.path.join(data_path, 'train', '*.jpeg'))
    files.sort()
    h5f = h5py.File('train.h5', 'w')
    train_num = 0
    lenFile =len(files)
    lenFile = 20
    for i in range(lenFile):
        img = cv2.imread(files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            Img = np.expand_dims(Img[:,:,0].copy(), 0)
            Img = np.float32(normalize(Img))
            patches = Im2Patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:,:,:,n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
    h5f.close()
    # val
    print('\nprocess validation data')
    files.clear()
    files = glob.glob(os.path.join(data_path, 'valid', '*.jpeg'))
    files.sort()
    h5f = h5py.File('val.h5', 'w')
    val_num = 0
    for i in range(len(files)):
        print("file: %s" % files[i])
        img = cv2.imread(files[i])
        img = np.expand_dims(img[:,:,0], 0)
        img = np.float32(normalize(img))
        h5f.create_dataset(str(val_num), data=img)
        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)

class Dataset(udata.Dataset):
    def __init__(self, train=True,transform = None):
        super(Dataset, self).__init__()
        self.transform = transform        
        self.train = train

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
        return data,label

    def normalize(self,data):
        return data/255.

    def swapaxies(self,img):
        img = img.swapaxes(0, 2)
        return img

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
    #label_list = np.zeros((file_number,), dtype=int)
    #img_data = []
    #com_type = np.dtype([('img_data',np.float),('label','i')])
    for i in range(len(sampled_files)):
        file_with_label = sampled_files[i]
        img = cv2.imread(file_with_label[0])
        label = dir_list.index(file_with_label[1])
        print(label)
        #label_list[i]=label
        #img_data.append(file_with_label[0]) 
        g = h5f.create_group(str(val_num))                        
        g.create_dataset('data', data=img)
        g.create_dataset('label', data=label,dtype=np.intp)      
    
        val_num += 1
   
    h5f.close()


def noise_generator (noise_type,image):
    """
    Generate noise to a given Image based on required noise type
    
    Input parameters:
        image: ndarray (input image data. It will be converted to float)
        
        noise_type: string
            'gauss'        Gaussian-distrituion based noise
            'poission'     Poission-distribution based noise
            's&p'          Salt and Pepper noise, 0 or 1
            'speckle'      Multiplicative noise using out = image + n*image
                           where n is uniform noise with specified mean & variance
    """
    row,col,ch= image.shape
    if noise_type == "gauss":       
        mean = 0.0
        var = 0.01
        sigma = var**0.5
        gauss = np.array(image.shape)
        gauss = np.random.normal(mean,sigma,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        noisy = image + gauss
        return noisy.astype('uint8')
    elif noise_type == "s&p":
        s_vs_p = 0.5
        amount = 0.004
        out = image
        # Generate Salt '1' noise
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
        out[coords] = 255
        # Generate Pepper '0' noise
        num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
        out[coords] = 0
        return out
    elif noise_type == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_type =="speckle":
        gauss = np.random.randn(row,col,ch)
        gauss = gauss.reshape(row,col,ch)        
        noisy = image + image * gauss
        return noisy
    else:
        return image

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


dir_list = ['CNV','DME','DRUSEN','NORMAL']
#dir_num = [1,2, 3,4 ]
tstlist = ['sdfsfe']
n = np.random.permutation(4)

label = dir_list.index('DME')
newlist=[tstlist,dir_list[0]]
print(newlist[1])

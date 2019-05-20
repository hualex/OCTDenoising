import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
import torch.cuda as cuda
import numpy as np
import time
import os

from torch.autograd import Variable
from matplotlib import pyplot as plt

# from function_utils import get_dataset
#from google.colab import drive
# drive.mount('/content/gdrive')


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print(
                f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(
                f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), 'checkpoint.pt')
        self.val_loss_min = val_loss


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Get a batch of training data

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


resnet18 = OCTResNet().cuda()

train_pathname = './gdrive/My Drive/OCT2017/train'
test_pathname = './gdrive/My Drive/OCT2017/test'
train_size = 1000
valid_size = 100
# Data loading code

mean_gray = 0.1307
stddev_gray = 0.3081

transform = transforms.Compose([transforms.Grayscale(),
                                transforms.CenterCrop(450),
                                transforms.ToTensor(),
                                transforms.Normalize(
                                    (mean_gray,), (stddev_gray,))
                                ])

total_oct_train = datasets.ImageFolder(train_pathname, transform=transform)
oct_train, oct_valid, _ = torch.utils.data.random_split(
    total_oct_train, [train_size, valid_size, len(total_oct_train)-train_size-valid_size])
oct_test = datasets.ImageFolder(test_pathname, transform=transform)

train_loader = torch.utils.data.DataLoader(
    oct_train, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

valid_loader = torch.utils.data.DataLoader(
    oct_valid, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)


resnet18 = OCTResNet().cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(resnet18.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(resnet18.parameters(), lr=0.0001)
es_resnet = EarlyStopping()

for epoch in range(1000):  # loop over the dataset multiple times

    s = time.time()
    running_loss = 0.0
    valid_loss = 0.0
    for inputs, labels in train_loader:
        # get the inputs

        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    for inputs, labels in valid_loader:

        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        outputs = resnet18(inputs)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
    print("epoche:", epoch)
    print("train loss:", running_loss/len(train_loader))
    valid_loss = valid_loss/len(valid_loader)
    print("valid loss:", valid_loss)
    print("elapsed time:", time.time()-s)
    es_resnet(valid_loss, resnet18)
    if es_resnet.early_stop:
        print("Early stopping")
        break


print('Finished Training')

resnet18.load_state_dict(torch.load('checkpoint.pt'))
dataiter = iter(valid_loader)
images, labels = dataiter.next()

# print images

imshow(torchvision.utils.make_grid(images))
print(labels.numpy())

img = Variable(images).cuda()
outputs = resnet18(img)
print(outputs.shape)
print(torch.max(outputs, 1).indices.cpu())


# test accuracy
test_loader = torch.utils.data.DataLoader(
    oct_test, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)
test_accuracy = 0.0
true_account = 0
for images, labels in test_loader:
    img = Variable(images).cuda()
    outputs = resnet18(img)
    labels = labels.cpu().numpy()
    out_labels = torch.max(outputs, 1).indices.cpu().detach().numpy()
    t_acc = np.sum(labels == out_labels)
    print(t_acc)
    true_account = true_account + t_acc

test_accuracy = true_account/len(oct_test)
print(test_accuracy)

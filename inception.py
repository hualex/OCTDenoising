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


class OCTInception(nn.Module):
    def __init__(self):
        super(OCTInception, self).__init__()

        self.OCTInception = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=1),
            torchvision.models.inception_v3(pretrained=True, aux_logits=False),
            nn.Linear(1000, 4))

    def forward(self, x):
        return self.OCTInception(x)


train_pathname = './gdrive/My Drive/OCT2017/train'
test_pathname = './gdrive/My Drive/OCT2017/test'
train_size = 1000
valid_size = 100
# Data loading code

mean_gray = 0.1307
stddev_gray = 0.3081

transform = transforms.Compose([transforms.Grayscale(),
                                transforms.CenterCrop(450),
                                transforms.Resize((299, 299)),
                                transforms.ToTensor()
                                ])

total_oct_train = datasets.ImageFolder(train_pathname, transform=transform)
oct_train, oct_valid, _ = torch.utils.data.random_split(
    total_oct_train, [train_size, valid_size, len(total_oct_train)-train_size-valid_size])
oct_test = datasets.ImageFolder(test_pathname, transform=transform)

train_loader = torch.utils.data.DataLoader(
    oct_train, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)

valid_loader = torch.utils.data.DataLoader(
    oct_valid, batch_size=16, shuffle=True, num_workers=2, pin_memory=True)


model = OCTInception().cuda()

criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)
optimizer = optim.Adam(model.parameters(), lr=0.0001)
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
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

    for inputs, labels in valid_loader:

        inputs = Variable(inputs).cuda()
        labels = Variable(labels).cuda()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
    print("epoche:", epoch)
    print("train loss:", running_loss/len(train_loader))
    valid_loss = valid_loss/len(valid_loader)
    print("valid loss:", valid_loss)
    print("elapsed time:", time.time()-s)
    es_resnet(valid_loss, model)
    if es_resnet.early_stop:
        print("Early stopping")
        break


print('Finished Training')

model.load_state_dict(torch.load('checkpoint.pt'))
dataiter = iter(valid_loader)
images, labels = dataiter.next()

# print images

imshow(torchvision.utils.make_grid(images))
print(labels.numpy())

img = Variable(images).cuda()
outputs = model(img)
print(outputs.shape)
print(torch.max(outputs, 1).indices.cpu())


# test accuracy
correct = 0
total = 0
with torch.no_grad():
    for images, labels in valid_loader:
        images = images.cuda()
        labels = labels.cuda()
        model.eval()
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        print(predicted.cpu().numpy())
        print(labels.cpu().numpy())
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 1000 test images: %d %%' % (
    100 * correct / total))

# test accuracy for each class
classes = total_oct_train.classes
class_correct = list(0. for i in range(len(classes)))
class_total = list(0. for i in range(len(classes)))

with torch.no_grad():
    for data in valid_loader:
        images, labels = data
        images = images.cuda()
        labels = labels.cuda()
        outputs = model(images)
        batch_len = len(labels)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()

        for i in range(batch_len):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(4):
    print('Accuracy of %5s : %2d %%' % (
        classes[i], 100 * class_correct[i] / class_total[i]))

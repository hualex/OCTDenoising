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


from torch.autograd import Variable
from matplotlib import pyplot as plt

"""
from google.colab import drive
drive.mount('/content/gdrive')
"""


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

class MnistResNet(nn.Module):
    def __init__(self):
        super(MnistResNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(1, 3,
                                     kernel_size=(7, 7),
                                     stride=(2, 2),
                                     padding=(3, 3), bias=False)
        self.resnet18 = torchvision.models.resnet18(pretrained=True)
        self.minst_resnet = torch.nn.Sequential(self.conv1, self.resnet18)

    def forward(self, x):
        return self.minst_resnet(x)


if __name__ == "__main__":

    # Data loading code
    """
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
    train_pathname = './gdrive/My Drive/OCT2017/train'
    test_pathname = './gdrive/My Drive/OCT2017/test'
    train_size = 100
    valid_size = 10


    """
    mean_gray = 0.1307
    stddev_gray = 0.3081

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((mean_gray,), (stddev_gray,))])

    mnist_train = datasets.MNIST(
        './data', train=True, download=True, transform=transform)
    mnist_valid = datasets.MNIST(
        './data', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(
        mnist_train, batch_size=20, shuffle=True, num_workers=2, pin_memory=True)

    valid_loader = torch.utils.data.DataLoader(
        mnist_valid, batch_size=20, shuffle=True, num_workers=2, pin_memory=True)

    resnet18 = MnistResNet().cuda()

    # inception = models.inception_v3()
    inputs, classes = next(iter(train_loader))

    # Make a grid from batch
    out = torchvision.utils.make_grid(inputs)

    #imshow(out, title=[x.numpy() for x in classes])

    # googlenet = models.googlenet()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(resnet18.parameters(), lr=0.001, momentum=0.9)
    es_resnet = EarlyStopping()
    for epoch in range(2):  # loop over the dataset multiple times

        running_loss = 0.0
        valid_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            # get the inputs
            inputs, labels = data
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
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
        for i, data in enumerate(valid_loader, 0):
            inputs, labels = data
            inputs = Variable(inputs).cuda()
            labels = Variable(labels).cuda()
            outputs = resnet18(inputs)
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
        valid_loss = valid_loss/len(valid_loader)
        es_resnet(valid_loss, resnet18)
        if es_resnet.early_stop:
            print("Early stopping")
            break

    print('Finished Training')

    dataiter = iter(valid_loader)
    images, labels = dataiter.next()

    # print images
    imshow(torchvision.utils.make_grid(images))
    print('GroundTruth: ', ' '.join('%5s' %
                                    classes[labels[j]] for j in range(20)))

    img = Variable(images).cuda()
    outputs = resnet18(img)
    print(torch.max(outputs, 1))
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

# Root directory for dataset
dataroot = "data/celeba"

# Number of workers for dataloader
workers = 2

# Batch size during training
batch_size = 128

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 64

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 64

# Size of feature maps in discriminator
ndf = 64

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparam for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.ImageFolder(root=dataroot,
                           transform=transforms.Compose([
                               transforms.Resize(image_size),
                               transforms.CenterCrop(image_size),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (
    torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8, 8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[
           :64], padding=2, normalize=True).cpu(), (1, 2, 0)))


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Reshape1(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 512, 7, 7)


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Reshape1(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), 128, 7, 7)


class GAN(torch.nn.Module):

    def __init__(self):
        super(GAN, self).__init__()

        self.generator = nn.Sequential(

            nn.Linear(100, 6272, bias=True),
            nn.BatchNorm1d(num_features=6272),
            nn.LeakyReLU(inplace=True, negative_slope=0.0001),
            Reshape1(),

            nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=(
                4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(inplace=True, negative_slope=0.0001),

            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(
                4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(inplace=True, negative_slope=0.0001),

            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=(
                4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(inplace=True, negative_slope=0.0001),

            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=(
                4, 4), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(inplace=True, negative_slope=0.0001),
            # nn.Dropout2d(p=0.2),

            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=(
                5, 5), stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(num_features=3),
            nn.LeakyReLU(inplace=True, negative_slope=0.0001),

            nn.ConvTranspose2d(in_channels=3, out_channels=1, kernel_size=(
                4, 4), stride=(2, 2), padding=1, bias=False),
            nn.Tanh()
        )

        self.discriminator = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, padding=1,
                      kernel_size=(3, 3), stride=(2, 2), bias=False),
            nn.BatchNorm2d(num_features=8),
            nn.LeakyReLU(inplace=True, negative_slope=0.0001),
            # nn.Dropout2d(p=0.2),

            nn.Conv2d(in_channels=8, out_channels=32, padding=1,
                      kernel_size=(3, 3), stride=(2, 2), bias=False),
            nn.BatchNorm2d(num_features=32),
            nn.LeakyReLU(inplace=True, negative_slope=0.0001),
            # nn.Dropout2d(p=0.2),

            Flatten(),

            nn.Linear(7*7*32, 1),
            # nn.Sigmoid()
            nn.Conv2d(in_channels=1000, out_channels=1,
                      kernel_size=(1, 1), stride=(1, 1), bias=False)

        )
        )

    def generator_forward(self, z):
        img=self.generator(z)
        return img

    def discriminator_forward(self, img):
        pred=model.discriminator(img)


# Create the generator
netG=Generator(ngpu).to(device)

# Handle multi-gpu if desired
if (device.type == 'cuda') and (ngpu > 1):
    netG=nn.DataParallel(netG, list(range(ngpu)))

# Apply the weights_init function to randomly initialize all weights
#  to mean=0, stdev=0.2.
netG.apply(weights_init)

# Print the model
print(netG)


# Training Loop

# Lists to keep track of progress
img_list=[]
G_losses=[]
D_losses=[]
iters=0

print("Starting Training Loop...")
# For each epoch
for epoch in range(num_epochs):
    # For each batch in the dataloader
    for i, data in enumerate(dataloader, 0):

        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # Train with all-real batch
        netD.zero_grad()
        # Format batch
        real_cpu=data[0].to(device)
        b_size=real_cpu.size(0)
        label=torch.full((b_size,), real_label, device = device)
        # Forward pass real batch through D
        output=netD(real_cpu).view(-1)
        # Calculate loss on all-real batch
        errD_real=criterion(output, label)
        # Calculate gradients for D in backward pass
        errD_real.backward()
        D_x=output.mean().item()

        # Train with all-fake batch
        # Generate batch of latent vectors
        noise=torch.randn(b_size, nz, 1, 1, device = device)
        # Generate fake image batch with G
        fake=netG(noise)
        label.fill_(fake_label)
        # Classify all fake batch with D
        output=netD(fake.detach()).view(-1)
        # Calculate D's loss on the all-fake batch
        errD_fake=criterion(output, label)
        # Calculate the gradients for this batch
        errD_fake.backward()
        D_G_z1=output.mean().item()
        # Add the gradients from the all-real and all-fake batches
        errD=errD_real + errD_fake
        # Update D
        optimizerD.step()

        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        netG.zero_grad()
        label.fill_(real_label)  # fake labels are real for generator cost
        # Since we just updated D, perform another forward pass of all-fake batch through D
        output=netD(fake).view(-1)
        # Calculate G's loss based on this output
        errG=criterion(output, label)
        # Calculate gradients for G
        errG.backward()
        D_G_z2=output.mean().item()
        # Update G
        optimizerG.step()

        # Output training stats
        if i % 50 == 0:
            print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                  % (epoch, num_epochs, i, len(dataloader),
                     errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

        # Save Losses for plotting later
        G_losses.append(errG.item())
        D_losses.append(errD.item())

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
            with torch.no_grad():
                fake=netG(fixed_noise).detach().cpu()
            img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

        iters += 1


plt.figure(figsize = (10, 5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(G_losses, label = "G")
plt.plot(D_losses, label = "D")
plt.xlabel("iterations")
plt.ylabel("Loss")
plt.legend()
plt.show()

fig=plt.figure(figsize = (8, 8))
plt.axis("off")
ims=[[plt.imshow(np.transpose(i, (1, 2, 0)), animated= True)]
       for i in img_list]
ani = animation.ArtistAnimation(
    fig, ims, interval=1000, repeat_delay=1000, blit=True)

HTML(ani.to_jshtml())


# Grab a batch of real images from the dataloader
real_batch = next(iter(dataloader))

# Plot the real images
plt.figure(figsize=(15, 15))
plt.subplot(1, 2, 1)
plt.axis("off")
plt.title("Real Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[
           :64], padding=5, normalize=True).cpu(), (1, 2, 0)))

# Plot the fake images from the last epoch
plt.subplot(1, 2, 2)
plt.axis("off")
plt.title("Fake Images")
plt.imshow(np.transpose(img_list[-1], (1, 2, 0)))
plt.show()
if __name__ == "__main__":
    torch.manual_seed(0)
    dataset_train_total = datasets.MNIST(
        root='./data', train=True, download=True, transform=transforms.ToTensor())
    dataset_train, dataset_val = torch.utils.data.random_split(
        dataset_train_total, [58800, 1200])
    dataset_test = datasets.MNIST(
        root='./data', train=False, download=True, transform=transforms.ToTensor())

    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, num_workers=2, pin_memory=True)
    dataloader_val = DataLoader(
        dataset_val, batch_size=batch_size, num_workers=2, pin_memory=True)
    dataloader_test = DataLoader(
        dataset_test, batch_size=batch_size, num_workers=2, pin_memory=True)

    train_loss = []
    valid_loss = []
    time_epoch = []

    # autoencoder.load_state_dict(torch.load('checkpoint.pt'))
    print(len(dataset_test))

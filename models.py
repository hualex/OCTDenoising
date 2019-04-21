import torch
import torch.nn as nn


class DenoisingAutoencoder(nn.Module):
    
    def __init__(self,depth = 3,image_channels=3):
    
        super(DenoisingAutoencoder, self).__init__()
        kernel_size =3
        padding =1
        encoder_layers = []
        decoder_layers = []
        channnel_num = [2**(depth-i+3) for i in range(depth)]
        
        #channel_size_list = [image_channels,12, 24, 48]
        channel_size_list = [image_channels] + channnel_num
        convlayer_numbers = len(channel_size_list)-1

        # Encoder laysers

        for i in range(convlayer_numbers):
            encoder_layers.append(nn.Conv2d(in_channels=channel_size_list[i], out_channels = channel_size_list[i+1],kernel_size=kernel_size,padding= padding,stride=1))
            #encoder_layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*encoder_layers)

        # Maxpooling layer

        mp_kernel_size = 2
        self.mp_layer = nn.MaxPool2d(mp_kernel_size,stride=2 ,return_indices=True)

        # Maxunpooling layer
         
        self.mup_layer = nn.MaxUnpool2d(mp_kernel_size,stride=2)

        # Decoder layers

        for j in range(convlayer_numbers):
            k = convlayer_numbers-j
            decoder_layers.append(nn.ConvTranspose2d(in_channels=channel_size_list[k], out_channels = channel_size_list[k-1],kernel_size=kernel_size,padding= padding,stride=1))
            decoder_layers.append(nn.ReLU(inplace=True))

        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self,x):

        x = self.encoder(x)
        #x,i = self.mp_layer(x)
        #x = self.mup_layer(x,i)
        y = self.decoder(x)

        return y


class DnCNN(nn.Module):
    def __init__(self, depth=10, n_channels=6, image_channels=3, kernel_size=3):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        layers = []
        layers.append(nn.Conv2d(in_channels=image_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(depth-2):
            layers.append(nn.Conv2d(in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(n_channels, eps=0.0001, momentum = 0.95))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=n_channels, out_channels=image_channels, kernel_size=kernel_size, padding=padding, bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        y = x
        out = self.dncnn(x)
        return y-out

#class REDNet(nn.modules):

#class MemNet(nn.modules):


class VAE(nn.Module):
    def __init__(self, image_size=56*56, h_dim=400, z_dim=20):
        super(VAE, self).__init__()
        
        self.fc1 = nn.Linear(image_size, h_dim)        
        self.fc2 = nn.Linear(h_dim, z_dim)
        self.fc3 = nn.Linear(h_dim, z_dim)
        self.fc4 = nn.Linear(z_dim, h_dim)
        self.fc5 = nn.Linear(h_dim, image_size)
        
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss_function(self,x_reconst,x):
        bce = nn.functional.binary_cross_entropy(x_reconst,x)
        kld = nn.functional.kl_div(x_reconst,x)
        return bce+kld

    
    def forward(self, x):
        h = nn.functional.relu(self.fc1(x))
        mu = nn.functional.relu(self.fc2(h))
        logvar = nn.functional.relu(self.fc3(h))
        z = self.reparameterize(mu, logvar)
        h = nn.functional.relu(self.fc4(z))
        x_reconst = nn.functional.sigmoid(self.fc5(h))
        return  x_reconst,mu, logvar 

if __name__ == "__main__":
    depth = 7
    channnel_num = [2**(depth-i+3) for i in range(depth)]
    image_channels = 3
    combine = [image_channels]+channnel_num
    print(combine)

    
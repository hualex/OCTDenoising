import torch
import torch.nn as nn

class DenoisingAutoencoder(nn.Module):
    
    def __init__(self):
    
        super(DenoisingAutoencoder, self).__init__()
        image_channels =3
        kernel_size =3
        padding =1
        encoder_layers = []
        decoder_layers = []
        channel_size_list = [image_channels,12, 24, 48]
        convlayer_numbers = len(channel_size_list)-1

        # Encoder laysers

        for i in range(convlayer_numbers):
            encoder_layers.append(nn.Conv2d(in_channels=channel_size_list[i], out_channels = channel_size_list[i+1],kernel_size=kernel_size,padding= padding))
            encoder_layers.append(nn.ReLU(inplace=True))

        self.encoder = nn.Sequential(*encoder_layers)

        # Maxpooling layer

        mp_kernel_size = 2
        self.mp_layer = nn.MaxPool2d(mp_kernel_size,return_indices=True)

        # Maxunpooling layer
         
        self.mup_layer = nn.MaxUnpool2d(mp_kernel_size)

        # Decoder layers

        for j in range(convlayer_numbers):
            k = convlayer_numbers-j
            decoder_layers.append(nn.ConvTranspose2d(in_channels=channel_size_list[k], out_channels = channel_size_list[k-1],kernel_size=kernel_size,padding= padding))
            decoder_layers.append(nn.ReLU(inplace=True))

        self.decoder = nn.Sequential(*decoder_layers)


    def forward(self,x):

        x = self.encoder(x)
        x,i = self.mp_layer(x)
        x = self.mup_layer(x,i)
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

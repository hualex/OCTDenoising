import torch
import torch.nn as nn


class convBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(convBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class downBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(downBlock, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            convBlock(in_ch, in_ch),
            convBlock(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class upBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(upBlock, self).__init__()
        self.up = nn.Sequential(
            convBlock(in_ch, out_ch),
            convBlock(out_ch, out_ch),
            nn.ConvTranspose3d(out_ch, out_ch, kernel_size=(
                1, 2, 2), stride=(1, 2, 2))
        )

    def forward(self, x1, x2):
        x = torch.cat([x2, x1], dim=1)
        y = self.up(x)
        return y


class bottomBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(bottomBlock, self).__init__()
        self.mpconv = nn.Sequential(
            downBlock(in_ch, out_ch),
            nn.ConvTranspose3d(out_ch, out_ch, kernel_size=(
                1, 2, 2), stride=(1, 2, 2))
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class outBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outBlock, self).__init__()
        self.conv_out = nn.Sequential(
            convBlock(in_ch, out_ch),
            convBlock(out_ch, out_ch),
            convBlock(out_ch, 3),
            convBlock(3, 1)
        )

    def forward(self, x1, x2):
        x = torch.cat([x1, x2], dim=1)
        y = self.conv_out(x)
        return y

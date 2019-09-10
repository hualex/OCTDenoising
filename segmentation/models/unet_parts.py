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


class DiceLoss(nn.Module):
    """Computes Dice Loss, which just 1 - DiceCoefficient described above.
    Additionally allows per-class weights to be provided.
    """

    def __init__(self, epsilon=1e-5, weight=None, ignore_index=None, sigmoid_normalization=True,
                 skip_last_target=False):
        super(DiceLoss, self).__init__()
        self.epsilon = epsilon
        self.register_buffer('weight', weight)
        self.ignore_index = ignore_index
        # The output from the network during training is assumed to be un-normalized probabilities and we would
        # like to normalize the logits. Since Dice (or soft Dice in this case) is usually used for binary data,
        # normalizing the channels with Sigmoid is the default choice even for multi-class segmentation problems.
        # However if one would like to apply Softmax in order to get the proper probability distribution from the
        # output, just specify sigmoid_normalization=False.
        if sigmoid_normalization:
            self.normalization = nn.Sigmoid()
        else:
            self.normalization = nn.Softmax(dim=1)
        # if True skip the last channel in the target
        self.skip_last_target = skip_last_target

    def forward(self, input, target):
        # get probabilities from logits
        input = self.normalization(input)
        if self.weight is not None:
            weight = Variable(self.weight, requires_grad=False)
        else:
            weight = None

        if self.skip_last_target:
            target = target[:, :-1, ...]

        per_channel_dice = compute_per_channel_dice(input, target, epsilon=self.epsilon, ignore_index=self.ignore_index,
                                                    weight=weight)
        # Average the Dice score across all channels/classes
        return torch.mean(1. - per_channel_dice)


def dice_loss(input, target):
    assert input.shape == target.shape
    numerator = torch.sum(torch.mul(input, target))
    denominator = torch.sum(input**2+target**2)
    return 1-2*numerator/denominator

import torch
import torch.nn as nn

from models.unet import UnetResnet50

class SEBlock(nn.Module):
    def __init__(self, in_ch, ratio=16):
        super(SEBlock, self).__init__()
        self.in_ch = in_ch
        self.ratio = ratio
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.gate = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // ratio, 1, 1),
            nn.ReLU(),
            nn.Conv2d(in_ch // ratio, in_ch, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        desc = self.avgpool(x)
        desc = self.gate(desc)
        return desc * x


class SEUnetResnet50(UnetResnet50):
    def __init__(self, num_classes=2, activation=None, pretrained=False):
        super(SEUnetResnet50, self).__init__(num_classes, activation, pretrained)
        self.down1 = nn.Sequential(self.down1, SEBlock(64))
        self.down2 = nn.Sequential(self.down2, SEBlock(256))
        self.down3 = nn.Sequential(self.down3, SEBlock(512))
        self.down4 = nn.Sequential(self.down4, SEBlock(1024))

        self.se_up4 = SEBlock(512)
        self.se_up3 = SEBlock(256)
        self.se_up2 = SEBlock(128)

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        up4 = self.up4(down4, down3)
        up4 = self.se_up4(up4)
        up3 = self.up3(up4, down2)
        up3 = self.se_up3(up3)
        up2 = self.up2(up3, down1)
        up2 = self.se_up2(up2)
        out = self.final(up2)
        if self.activation == 'sigmoid':
            return torch.sigmoid(out)
        return out



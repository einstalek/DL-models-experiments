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

        self.up4 = nn.Sequential(self.up4, SEBlock(1024))
        self.up3 = nn.Sequential(self.up3, SEBlock(512))
        self.up2 = nn.Sequential(self.up2, SEBlock(256))

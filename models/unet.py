import torch
import torch.nn as nn
import torchvision


class UnetConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.2):
        super(UnetConvBlock, self).__init__()
        self.dropout = dropout
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.bn1(x)
        x = nn.Dropout(self.dropout)(x)
        x = self.conv2(x)
        x = nn.ReLU()(x)
        x = self.bn2(x)
        return x


class UnetBlockUp(nn.Module):
    def __init__(self, in_ch, prev_ch=None, up_type='upsample'):
        super(UnetBlockUp, self).__init__()
        out_ch = in_ch // 2
        if prev_ch is None:
            prev_ch = out_ch
        if up_type == 'upsample':
            self.upconv = nn.Sequential(
                nn.Upsample(mode='bilinear', scale_factor=2),
                nn.Conv2d(in_ch, out_ch, 1),
            )
        else:
            self.upconv = nn.ConvTranspose2d(in_ch, out_ch, 2, 2)
        self.conv = UnetConvBlock(out_ch + prev_ch, out_ch, dropout=0)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat([x, skip], 1)
        x = self.conv(x)
        return x


class Unet(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2, activation=None):
        super(Unet, self).__init__()
        self.num_classes = num_classes
        self.activation = activation
        self.dropout = dropout

        self.pool = nn.MaxPool2d(2)
        self.down1 = UnetConvBlock(3, 64, dropout)
        self.down2 = UnetConvBlock(64, 128, dropout)
        self.down3 = UnetConvBlock(128, 256, dropout)
        self.down4 = UnetConvBlock(256, 512, dropout)
        self.down5 = UnetConvBlock(512, 1024, dropout)

        self.up5 = UnetBlockUp(1024)
        self.up4 = UnetBlockUp(512)
        self.up3 = UnetBlockUp(256)
        self.up2 = UnetBlockUp(128)
        self.final = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        down1 = self.down1(x)
        x = self.pool(down1)
        down2 = self.down2(x)
        x = self.pool(down2)
        down3 = self.down3(x)
        x = self.pool(down3)
        down4 = self.down4(x)
        x = self.pool(down4)
        down5 = self.down5(x)

        up5 = self.up5(down5, down4)
        up4 = self.up4(up5, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        out = self.final(up2)
        if self.activation == 'sigmoid':
            out = torch.sigmoid(out)
        return out


class UnetResnet50(nn.Module):
    def __init__(self, num_classes=2, activation=None, pretrained=False):
        super(UnetResnet50, self).__init__()
        base_model = torchvision.models.resnet50(pretrained=pretrained)
        self.activation = activation
        self.down1 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu
        )
        self.down2 = nn.Sequential(
            base_model.maxpool,
            base_model.layer1,
        )
        self.down3 = base_model.layer2
        self.down4 = base_model.layer3

        self.up4 = UnetBlockUp(1024)
        self.up3 = UnetBlockUp(512)
        self.up2 = UnetBlockUp(256, prev_ch=64)

        self.final = nn.Sequential(
            UnetConvBlock(128, 32),
            nn.Conv2d(32, num_classes, 1)
        )

    def forward(self, x):
        down1 = self.down1(x)
        down2 = self.down2(down1)
        down3 = self.down3(down2)
        down4 = self.down4(down3)

        up4 = self.up4(down4, down3)
        up3 = self.up3(up4, down2)
        up2 = self.up2(up3, down1)
        out = self.final(up2)
        if self.activation == 'sigmoid':
            return torch.sigmoid(out)
        return out

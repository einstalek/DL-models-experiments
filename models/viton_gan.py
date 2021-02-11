import torch
import torch.nn as nn
import torch.nn.functional as F
import re


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()


class Discriminator(nn.Module):
    def __init__(self, in_dim=6, d=64):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, d, 4, 2, 1)
        self.conv2 = nn.Conv2d(d, d * 2, 4, 2, 1)
        self.conv2_bn = nn.BatchNorm2d(d * 2)
        self.conv3 = nn.Conv2d(d * 2, d * 4, 4, 2, 1)
        self.conv3_bn = nn.BatchNorm2d(d * 4)
        self.conv4 = nn.Conv2d(d * 4, d * 8, 4, 2, 1)
        self.conv4_bn = nn.BatchNorm2d(d * 8)
        self.conv5 = nn.Conv2d(d * 8, d * 16, 4, 1, 1)
        self.conv5_bn = nn.BatchNorm2d(d * 16)
        self.conv6 = nn.Conv2d(d * 16, 1, 4, 1, 1)
        self.relu = nn.LeakyReLU(0.2)

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, label):
        x = torch.cat([input, label], 1)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2_bn(self.conv2(x)))
        x = self.relu(self.conv3_bn(self.conv3(x)))
        x = self.relu(self.conv4_bn(self.conv4(x)))
        x = self.relu(self.conv5_bn(self.conv5(x)))
        x = self.conv6(x).sigmoid()
        return x


class SPADE(nn.Module):
    def __init__(self, config_text, norm_nc, label_nc):
        super().__init__()
        assert config_text.startswith('spade')
        parsed = re.search('spade(\D+)(\d)x\d', config_text)
        param_free_norm_type = str(parsed.group(1))
        ks = int(parsed.group(2))

        if param_free_norm_type == 'instance':
            self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        elif param_free_norm_type == 'batch':
            self.param_free_norm = nn.BatchNorm2d(norm_nc, affine=False)
        else:
            raise ValueError('%s is not a recognized param-free norm type in SPADE'
                             % param_free_norm_type)
        nhidden = 128

        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, nhidden, kernel_size=ks, padding=pw),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x, segmap):
        # Part 1. generate parameter-free normalized activations
        normalized = self.param_free_norm(x)

        # Part 2. produce scaling and bias conditioned on semantic map
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class SPADEResidBlock(nn.Module):
    def __init__(self, in_ch, out_ch, label_ch, hid_dim=None):
        super(SPADEResidBlock, self).__init__()
        if hid_dim is None:
            hid_dim = out_ch
        self.spade1 = SPADE('spadebatch3x3', in_ch, label_ch)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(in_ch, hid_dim, 3, padding=1, bias=False)
        self.spade2 = SPADE('spadebatch3x3', hid_dim, label_ch)
        self.conv2 = nn.Conv2d(hid_dim, out_ch, 3, padding=1, bias=False)
        self.skip_conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.skip_spade = SPADE('spadebatch3x3', in_ch, label_ch)

    def forward(self, x, segm):
        inp = x
        x = self.spade1(x, segm)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.spade2(x, segm)
        x = self.relu(x)
        x = self.conv2(x)
        skip = self.skip_spade(inp, segm)
        skip = self.relu(skip)
        skip = self.skip_conv(skip)
        return x + skip


def sn_conv2d(in_ch, out_ch, ksize=1, strides=1, padding=0, bias=True):
    return nn.utils.spectral_norm(nn.Conv2d(in_ch, out_ch, ksize,
                                            stride=strides, padding=padding,
                                            bias=bias))

class UpBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv = sn_conv2d(in_ch, out_ch, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
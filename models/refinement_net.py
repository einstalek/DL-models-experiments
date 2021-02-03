import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(1, features, 1, 1), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, features, 1, 1), requires_grad=True)
        self.eps = eps

    def forward(self, x):
        batch_size = x.size()[0]
        x_view = x.view(batch_size, -1)
        mean = x_view.mean(1).view(batch_size, 1, 1, 1)
        std = x_view.std(1).view(batch_size, 1, 1, 1)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ConvModule(nn.Module):
    def __init__(self, channels_in, channels_out, final_module=False, final_dim=3):
        super(ConvModule, self).__init__()
        self.conv1 = nn.Conv2d(channels_in, channels_out, (3, 3), (1, 1), (1, 1), bias=False)
        self.ln1 = LayerNorm(channels_out)
        self.lrelu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(channels_out, channels_out, (3, 3), (1, 1), (1, 1), bias=False)

        self.final_module = final_module
        if not self.final_module:
            self.ln2 = LayerNorm(channels_out)
            self.lrelu2 = nn.LeakyReLU(0.1)
        else:
            self.conv3 = nn.Conv2d(channels_out, final_dim, (1, 1), (1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.ln1(x)
        x = self.lrelu1(x)
        x = self.conv2(x)
        if not self.final_module:
            x = self.ln2(x)
            x = self.lrelu2(x)
        else:
            x = self.conv3(x)
        return x


class RefineNet(nn.Module):
    def __init__(self, in_dim, out_dim, num_modules=4,
                 dims=(512, 256, 128, 64, 32)):
        super(RefineNet, self).__init__()
        self.layers = []
        _inp = in_dim
        for i in range(num_modules):
            self.layers.append(ConvModule(_inp, dims[i]))
            _inp = dims[i]
        self.layers.append(ConvModule(dims[-2], dims[-1],
                                      final_module=True, final_dim=out_dim))
        self.layers = nn.ModuleList(self.layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


if __name__ == '__main__':
    net = RefineNet(3, 3)
    print(sum(p.numel() for p in net.parameters()))
    x = torch.randn(1, 3, 256, 192)
    out = net(x)
    print(out.shape)

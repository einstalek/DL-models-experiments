import torch
from torch import nn
from torchvision.models import vgg
from collections import namedtuple

LossOutput = namedtuple("LossOutput", ["relu1_2",
                                       "relu2_2",
                                       "relu3_2",
                                       "relu4_2",
                                       'relu5_2'])


class LossNetwork(torch.nn.Module):
    def __init__(self, vgg_model, device='cuda'):
        super(LossNetwork, self).__init__()
        self.vgg_layers = vgg_model.features
        self.layer_name_mapping = {
            '3': "relu1_2",
            '8': "relu2_2",
            '13': 'relu3_2',
            '22': 'relu4_2',
            '31': 'relu5_2',
        }
        self.device = device

    def forward(self, x, refinement=False):
        output = {}
        for name, module in self.vgg_layers._modules.items():
            with torch.no_grad():
                x = module(x)
            if name in self.layer_name_mapping:
                if refinement and int(name) < 13:
                    # don't use lower layers output during refinement
                    output[self.layer_name_mapping[name]] = torch.zeros(1).to(self.device)
                else:
                    output[self.layer_name_mapping[name]] = x
        return LossOutput(**output)


class PerceptualLoss:
    def __init__(self, device='cuda', crit=None):
        self.device = device
        self.vgg_model = vgg.vgg19(pretrained=True).eval().to(device)
        self.loss_model = LossNetwork(self.vgg_model, device=device)
        self.crit = crit
        if self.crit is None:
            self.crit = nn.SmoothL1Loss()

    def __call__(self, gt, out, refinement=False):
        """
        :param gt: target image in range (-1, 1)
        :param out: produced image in range(-1, 1)
        :return:
        """
        mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None].to(self.device)
        std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None].to(self.device)
        gt = (gt + 1.) / 2.
        gt = (gt - mean) / std
        out = (out + 1.) / 2.
        out = (out - mean) / std
        feats1 = self.loss_model(gt, refinement=refinement)
        feats2 = self.loss_model(out, refinement=refinement)
        loss = 0.
        for (x, y) in zip(feats1, feats2):
            loss += self.crit(x, y)
        return loss

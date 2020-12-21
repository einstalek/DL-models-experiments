import torch
import torchvision
import torch.nn as nn

from models.anchors import generate_anchor_boxes

class FPN(nn.Module):
    def __init__(self, *sizes, out_ch):
        super(FPN, self).__init__()
        self.sizes = sizes
        self.out_ch = out_ch

        self.upsample = nn.Upsample(scale_factor=2)
        self.side_conv5 = nn.Conv2d(sizes[-1], out_ch, 1)
        self.side_conv4 = nn.Conv2d(sizes[-2], out_ch, 1)
        self.merge_conv4 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.side_conv3 = nn.Conv2d(sizes[-3], out_ch, 1)
        self.merge_conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    def forward(self, *fmaps):
        C3, C4, C5 = fmaps
        P5 = self.side_conv5(C5)
        P4 = self.merge_conv4(self.side_conv4(C4) + self.upsample(P5))
        P3 = self.merge_conv3(self.side_conv3(C3) + self.upsample(P4))
        return P3, P4, P5


class Resnet50(nn.Module):
    def __init__(self, pretrained=False):
        super(Resnet50, self).__init__()
        base_model = torchvision.models.resnet50(pretrained=pretrained)
        self.layer1 = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool,
            base_model.layer1
        )
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

    def forward(self, x):
        x = self.layer1(x)
        C3 = self.layer2(x)
        C4 = self.layer3(C3)
        C5 = self.layer4(C4)
        return C3, C4, C5


class RetinaConvHead(nn.Module):
    def __init__(self, out_ch, in_ch=256, activatiton=None):
        super(RetinaConvHead, self).__init__()
        self.conv = [nn.Conv2d(in_ch, in_ch, 3, padding=1) for _ in range(4)]
        self.bn = [nn.BatchNorm2d(in_ch) for _ in range(4)]
        self.relu = nn.ReLU()
        self.final = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.activatiton = activatiton

    def forward(self, x):
        for i in range(4):
            x = self.conv[i](x)
            x = self.relu(x)
            x = self.bn[i](x)
        x = self.final(x)
        if self.activatiton == 'sigmoid':
            x = x.sigmoid()
        return x


class RetinaNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=2,
                 img_size=256, scales=(0.5,), ratios=(1, 0.5, 2)):
        super(RetinaNet, self).__init__()
        self.backbone = Resnet50(pretrained)
        fmap_sizes = (512, 1024, 2048)
        anchors = len(ratios) * len(scales)
        self.fpn = FPN(*fmap_sizes, out_ch=256)
        self.class_subnet = RetinaConvHead(num_classes * anchors, activatiton="sigmoid")
        self.regr_subnet = RetinaConvHead(4 * anchors)
        self.anchors = generate_anchor_boxes(fmap_sizes, 8, img_size, scales, ratios)

    def forward(self, x):
        fmaps = self.backbone(x)
        fpn_maps = self.fpn(*fmaps)
        class_out = []
        regr_out = []
        for fmap in fpn_maps:
            class_out.append(self.class_subnet(fmap))
            regr_out.append(self.regr_subnet(fmap))
        return class_out, regr_out


if __name__ == "__main__":
    # retina = RetinaNet(num_classes=1)
    # inp = torch.zeros(1, 3, 256, 256)
    # out = retina(inp)
    # print([x.shape for x in out[0]])
    inp = torch.rand(4, 3, 32, 32)
    x = inp.view(4, -1, 3)
    x = torch.max(x, -1)[0]
    x = torch.topk(x, 10, 1)[0]
    print(x)



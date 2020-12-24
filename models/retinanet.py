import torch
import torchvision
import torch.nn as nn
from torchvision import ops

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
    def __init__(self, out_ch, in_ch=256):
        super(RetinaConvHead, self).__init__()
        self.conv = nn.ModuleList([nn.Conv2d(in_ch, in_ch, 3, padding=1) for _ in range(4)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(in_ch) for _ in range(4)])
        self.relu = nn.ReLU()
        self.final = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def forward(self, x):
        for i in range(4):
            x = self.conv[i](x)
            x = self.relu(x)
            x = self.bn[i](x)
        x = self.final(x)
        return x


class RetinaNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=2,
                 img_size=256, scales=(0.5, 0.25), ratios=(1, 0.5, 2)):
        super(RetinaNet, self).__init__()
        self.backbone = Resnet50(pretrained)
        fmap_sizes = (32, 16, 8)
        fmap_channels = (512, 1024, 2048)
        self.anchors_num = len(ratios) * len(scales)
        self.fpn = FPN(*fmap_channels, out_ch=256)
        self.class_subnet = RetinaConvHead(num_classes * self.anchors_num)
        self.regr_subnet = RetinaConvHead(4 * self.anchors_num)
        self.anchors = generate_anchor_boxes(fmap_sizes, 8, img_size, scales, ratios, mode="xywh")
        self.num_classes = num_classes

    def forward(self, x):
        """
        :return:
            -- class logits, [B, A, K]
            -- regression  output, [B, A, 4]
        """
        bsize = x.size(0)
        fmaps = self.backbone(x)
        fpn_maps = self.fpn(*fmaps)
        class_out = []
        regr_out = []
        for fmap in fpn_maps:
            class_out.append(self.class_subnet(fmap).view(bsize, -1, self.num_classes))
            regr_out.append(self.regr_subnet(fmap).view(bsize, -1, 4))
        return torch.cat(class_out, 1), torch.cat(regr_out, 1)

    def inference(self, cls_out, regr_out,
                  max_iou=0.5, min_conf=0.2, k=50,
                  xy_std=0.1, wh_std=0.2):
        """
        :param cls_out: [B, A, K]
        :param regr_out: [B, A, 4]
        :return:
            -- boxes, [k, 4], xywh
            -- scores
        """
        boxes = []
        scores = []
        conf, pred_labels = cls_out.sigmoid().max(-1)  # (B, A, 1)
        bsize = cls_out.size(0)
        for i in range(bsize):
            filter_mask = torch.where(conf[i] > min_conf)
            top_boxes = self.anchors[filter_mask]  # (N, 4)
            top_conf = conf[i][filter_mask]  # (N,)
            top_regr = regr_out[i][filter_mask]  # (N, 4)
            pick_ids = ops.nms(ops.box_convert(top_boxes, "xywh", "xyxy"),
                               top_conf, max_iou)
            picked_boxes = top_boxes[pick_ids][:k]  # (k, 4)
            picked_conf = top_conf[pick_ids][:k]  # (k,)
            picked_regr = top_regr[pick_ids][:k]  # (k, 4)
            xy = picked_boxes[:, 2:] * xy_std * picked_regr[:, :2] + picked_boxes[:, :2]
            wh = picked_regr[:, 2:].exp() * wh_std * picked_boxes[:, 2:]
            boxes.append(torch.cat([xy, wh], -1))
            scores.append(picked_conf)
        return boxes, scores





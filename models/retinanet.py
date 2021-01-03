import torch
import torchvision
import torch.nn as nn
from torchvision import ops
import tqdm
import numpy as np

from datasets.utils import encode_batch
from models.anchors import generate_anchor_boxes


class FPN(nn.Module):
    def __init__(self, *sizes, out_ch):
        super(FPN, self).__init__()
        self.sizes = sizes  # (C3, C4, C5)
        self.out_ch = out_ch

        self.side_conv5 = nn.Conv2d(sizes[-1], out_ch, 1)
        self._init(self.side_conv5)
        self.bn5 = nn.BatchNorm2d(sizes[-1], affine=False)
        self.upsample5 = nn.Upsample(scale_factor=2)
        self.merge_conv5 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self._init(self.merge_conv5)
        self.side_conv4 = nn.Conv2d(sizes[-2], out_ch, 1)
        self._init(self.side_conv4)
        self.bn4 = nn.BatchNorm2d(sizes[-2], affine=False)
        self.upsample4 = nn.Upsample(scale_factor=2)
        self.merge_conv4 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self._init(self.merge_conv4)
        self.side_conv3 = nn.Conv2d(sizes[-3], out_ch, 1)
        self._init(self.side_conv3)
        self.bn3 = nn.BatchNorm2d(sizes[-3], affine=False)
        self.upsample3 = nn.Upsample(scale_factor=2)
        self.merge_conv3 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self._init(self.merge_conv3)
        self.conv6 = nn.Conv2d(sizes[-1], out_ch, kernel_size=3, stride=2, padding=1)
        self._init(self.conv6)
        self.bn6 = nn.BatchNorm2d(sizes[-1], affine=False)

    def _init(self, layer):
        torch.nn.init.xavier_normal_(layer.weight)
        torch.nn.init.zeros_(layer.bias)

    def forward(self, *fmaps):
        C3, C4, C5 = fmaps

        P5 = self.side_conv5(self.bn5(C5))
        P5_upsampled = self.upsample5(P5)
        P5 = self.merge_conv5(P5)

        P4 = self.side_conv4(self.bn4(C4)) + P5_upsampled
        P4_upsampled = self.upsample4(P4)
        P4 = self.merge_conv4(P4)

        P3 = self.merge_conv3(self.side_conv3(self.bn3(C3)) + P4_upsampled)
        P6 = self.conv6(self.bn6(C5))
        return P3, P4, P5, P6


class Resnet50(nn.Module):
    def __init__(self, pretrained=True):
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
        self.conv = nn.ModuleList([self._init(nn.Conv2d(in_ch, in_ch, 3, padding=1)) for _ in range(4)])
        self.bn = nn.ModuleList([nn.BatchNorm2d(in_ch) for _ in range(4)])
        self.relu = nn.ReLU()
        self.final = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    def _init(self, layer):
        torch.nn.init.zeros_(layer.bias)
        torch.nn.init.normal_(layer.weight, 0, 1e-2)
        return layer

    def forward(self, x):
        for i in range(4):
            x = self.conv[i](x)
            x = self.relu(x)
            x = self.bn[i](x)
        x = self.final(x)  # (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        return x.contiguous()


class RetinaNet(nn.Module):
    def __init__(self, pretrained=False, num_classes=2, img_size=256,
                 scales=(2**0, 2**1/3, 2**2/3), ratios=(1, 0.5, 2), subsample=2.5):
        super(RetinaNet, self).__init__()
        self.backbone = Resnet50(pretrained)
        fmap_sizes = (32, 16, 8, 4)
        fmap_channels = (512, 1024, 2048)
        self.anchors_num = len(ratios) * len(scales)
        self.fpn = FPN(*fmap_channels, out_ch=256)
        self.class_subnet = RetinaConvHead(num_classes * self.anchors_num)
        self.regr_subnet = RetinaConvHead(4 * self.anchors_num)
        self.anchors = generate_anchor_boxes(fmap_sizes, subsample, img_size, scales, ratios, mode="cxcywh").float()
        self.num_classes = num_classes
        self.img_size = img_size
        self.regr_weight = 1.
        self.cls_weight = 1.
        # Initialize final layers in subnets
        prior = 1e-2
        self.class_subnet.final.bias.data.fill_(-np.log((1-prior) / prior))
        self.regr_subnet.final.bias.data.fill_(0.)
        torch.nn.init.normal_(self.class_subnet.final.weight, 0, 1e-2)
        torch.nn.init.normal_(self.regr_subnet.final.weight, 0, 1e-2)

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
                  max_iou=0.5, min_conf=0.05, min_size=10, k=50,
                  xy_std=0.1, wh_std=0.2):
        """

        :param min_size: min size of a box
        :param cls_out: (B, A, K)
        :param regr_out:
        :param max_iou:
        :param min_conf:
        :param k:
        :param xy_std:
        :param wh_std:
        :return:
            -- boxes
            -- scores
            -- labels
        """
        boxes = []
        scores = []
        labels = []
        conf, pred_labels = cls_out.sigmoid().max(-1)  # (B, A, 1)
        bsize = cls_out.size(0)
        for i in range(bsize):
            filter_mask = torch.where(conf[i] > min_conf)
            top_boxes = self.anchors[filter_mask]  # (N, 4)
            top_conf = conf[i][filter_mask]  # (N,)
            top_regr = regr_out[i][filter_mask]  # (N, 4)
            top_labels = pred_labels[i][filter_mask]
            pick_ids = ops.nms(ops.box_convert(top_boxes, "cxcywh", "xyxy"), top_conf, max_iou)
            picked_boxes = top_boxes[pick_ids][:k]  # (k, 4)
            picked_regr = top_regr[pick_ids][:k]  # (k, 4)
            picked_scores, picked_labels = top_conf[pick_ids][:k], top_labels[pick_ids][:k]
            # gt_xy = A_wh * s_xy * dxdy + A_xy
            xy = picked_boxes[:, 2:] * xy_std * picked_regr[:, :2] + picked_boxes[:, :2]
            # gt_wh = A_wh * exp(dwdh * s_wh)
            wh = (picked_regr[:, 2:] * wh_std).exp() * picked_boxes[:, 2:]
            box = torch.cat([xy, wh], -1)
            box_xyxy = ops.box_convert(box, "cxcywh", "xyxy")
            box_xyxy = ops.clip_boxes_to_image(box_xyxy, (self.img_size, self.img_size))
            keep_ids = ops.remove_small_boxes(box_xyxy, min_size)
            box_xyxy = box_xyxy[keep_ids]
            box = ops.box_convert(box_xyxy, "xyxy", "cxcywh")
            boxes.append(box)
            scores.append(picked_scores[keep_ids])
            labels.append(picked_labels[keep_ids])
        return boxes, labels, scores


def train_single_epoch(model: RetinaNet, optimizer, dataloader,
                       cls_crit, reg_crit, epoch,
                       postfix_dict={}, device="cpu"):
    model.train()
    anchors = model.anchors
    total_step = len(dataloader)
    total_loss = 0
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step, position=0, leave=False)
    for i, (images, boxes, labels) in tbar:
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        cls_target, reg_target = encode_batch(anchors, boxes, labels, 2)
        cls_target = cls_target.to(device)
        reg_target = reg_target.to(device)

        if (cls_target >= 0).sum(1).min() < 1:
            continue

        cls_out, reg_out = model(images)
        cls_loss = cls_crit(cls_out, cls_target)
        regr_loss = (reg_crit(reg_out, reg_target) * (cls_target >= 0)).mean()
        loss = model.cls_weight * cls_loss + model.regr_weight * regr_loss
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        postfix_dict['train/loss'] = loss.item()

        f_epoch = epoch + i / total_step
        desc = '{:5s}'.format('train')
        desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)
    postfix_dict['train/loss'] = total_loss / len(dataloader)


def eval_single_batch(boxes, gt_boxes):
    """
    :param boxes: List of (B, tempK, 4)
    :param gt_boxes: Tensor (B, tempN, 4)
    :return:
    """
    bsize = gt_boxes.size(0)
    total_iou = 0.
    for i in range(bsize):
        pred = ops.box_convert(boxes[i], "cxcywh", "xyxy")
        gt = ops.box_convert(gt_boxes[i], "cxcywh", "xyxy")
        iou = ops.box_iou(gt, pred).mean().item()
        if np.isnan(iou):
            continue
        total_iou += iou
    return total_iou / bsize


def evaluale_single_epoch(model: RetinaNet, dataloader, cls_crit, reg_crit, epoch,
                          postfix_dict={}, k=10, device='cpu'):
    model.eval()
    total_step = len(dataloader)
    total_iou = 0.
    tbar = tqdm.tqdm(enumerate(dataloader), total=total_step, position=0, leave=False)
    for i, (images, boxes, labels) in tbar:
        images = images.to(device)
        boxes = boxes.to(device)  # (B, _, 4)
        labels = labels.to(device)
        cls_target, reg_target = encode_batch(model.anchors, boxes, labels, 2)
        cls_target = cls_target.to(device)
        reg_target = reg_target.to(device)

        if (cls_target >= 0).sum(1).min() < 1:
            continue

        cls_out, regr_out = model(images)
        cls_loss = cls_crit(cls_out, cls_target)
        regr_loss = (reg_crit(regr_out, reg_target) * (cls_target >= 0)).mean()
        loss = model.cls_weight * cls_loss + model.regr_weight * regr_loss
        postfix_dict['val/loss'] = loss.item()

        pred_boxes, pred_scores, pred_labels = model.inference(cls_out, regr_out, k=k)
        eval_single_batch(pred_boxes, boxes)

        iou = eval_single_batch(pred_boxes, boxes)
        total_iou += iou
        postfix_dict['val/iou'] = iou

        f_epoch = epoch + i / total_step
        desc = '{:5s}'.format('val')
        desc += ', {:06d}/{:06d}, {:.2f} epoch'.format(i, total_step, f_epoch)
        tbar.set_description(desc)
        tbar.set_postfix(**postfix_dict)
    postfix_dict['val/iou'] = total_iou / len(dataloader)


if __name__ == "__main__":
    images = torch.rand(8, 3, 256, 256)
    boxes = torch.from_numpy(np.array([[98.8713, 126.6131, 147.6946, 149.5331] * 8,
                                       [89.0334, 132.7554, 152.6815, 152.6237] * 8])).view(8, -1, 4)
    retina = RetinaNet()
    cls_out, regr_out = retina(images)
    # pboxes, labels, scores = retina.inference(cls_out, regr_out)
    # print(eval_single_batch(pboxes, boxes))


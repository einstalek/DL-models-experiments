from xml.etree import ElementTree as ET

import numpy as np
import torch
from torchvision import ops
from models.anchors import generate_anchor_boxes


def match_anchors_with_boxes(anchors, boxes, match_iou=0.5, ignore_iou=0.4):
    """
    :param anchors: [A, 4], xywh
    :param boxes: [N, 4], xywh
    :return:
        - matched_box_idx, [A], index of gt_box matched for each anchor
        - positive anchors bool  mask
        - ignore anchors bool  mask
        - negative anchors bool  mask
    """
    iou = ops.box_iou(ops.box_convert(anchors, "cxcywh", "xyxy"),
                      ops.box_convert(boxes, "cxcywh", "xyxy"))  # AxN
    vals, matched_box_idx = torch.max(iou, 1)
    positive = vals >= match_iou
    ignore = torch.logical_and(vals >= ignore_iou, vals < match_iou)
    negative = vals < ignore_iou
    return matched_box_idx, positive, ignore, negative


def get_targets(anchors, gt_boxes, cls_labels, num_classes,
                xy_std=0.1, wh_std=0.2,
                negative_index=-1, ignore_index=-2):
    """
    :param ignore_index:
    :param negative_index:
    :param anchors: [A, 4], "cxcywh"
    :param gt_boxes: [N, 4], "cxcywh"
    :param cls_labels: [N], class gt labels
    :param box_scale:
    :param xy_std:
    :param wh_std:
    :return:
        - classification target with classes, (A, K)
        - regression target for boxes, (A, 4)
    """
    matched_box_idx, positive, ignore, negative = match_anchors_with_boxes(anchors, gt_boxes)
    collected_gt_boxes = gt_boxes[matched_box_idx]
    regr_target = torch.cat([
        (collected_gt_boxes[:, :2] - anchors[:, :2]) / anchors[:, 2:] / xy_std,  # (xy_gt - xy_a) / wh_a / std_xy
        torch.log(collected_gt_boxes[:, 2:] / anchors[:, 2:]) / wh_std,  # log(wh_gt / wh_a) / std_wh
    ], 1).float()
    cls_target = cls_labels[matched_box_idx]
    A_num = cls_target.size(0)
    one_hot = torch.zeros(A_num, num_classes)
    one_hot[torch.arange(A_num), cls_target] = 1
    one_hot[torch.where(ignore)] = ignore_index
    one_hot[torch.where(negative)] = negative_index
    return one_hot, regr_target


def parse_xml(fp):
    obj = {}
    tree = ET.parse(fp)
    root = tree.getroot()
    for x in root:
        if x.tag == 'object':
            for xx in x:
                if xx.tag == 'name':
                    obj['name'] = xx.text
                elif xx.tag == 'bndbox':
                    box = []
                    for xxx in xx:
                        box.append(float(xxx.text))
                    obj['bbox'] = box
                elif xx.tag == 'occluded':
                    obj['occluded'] = int(xx.text)
                elif xx.tag == 'difficult':
                    obj['difficult'] = int(xx.text)
        elif x.tag == 'filename':
            obj['filename'] = x.text
        elif x.tag == 'fmap_size':
            size =  []
            for xx in x:
                size.append(int(xx.text))
            obj['fmap_size'] = size
    return obj


def encode_batch(anchors, boxes, labels, num_classes):
    """
    :param anchors: [A, 4]
    :param boxes: [B, N, 4], xywh
    :param labels: [B, N]
    :return:
    """
    cls_target, regr_target = [], []
    batch = boxes.size(0)
    for i in range(batch):
        cls, regr = get_targets(anchors, boxes[i], labels[i], num_classes=num_classes)
        cls_target.append(cls[None])
        regr_target.append(regr[None])
    return torch.cat(cls_target), torch.cat(regr_target)


if __name__ == "__main__":
    anchors = generate_anchor_boxes((32, 16, 8), 8, 256, (0.5,), (1, 0.5, 2), mode="xywh")
    boxes = torch.from_numpy(np.array([[10, 20, 120, 220],
                                       [120, 120, 250, 190],
                                       [0, 0, 50, 50]])).view(1, 3, 4)
    labels = torch.from_numpy(np.array([0, 1, 2])).view(1, 3)
    targets = encode_batch(anchors, boxes, labels, 3)
    print(targets[0])

import torch
import numpy as np
from models.retinanet import RetinaNet
from losses.losses import FocalLoss
import torch.nn as nn
from datasets.utils import encode_batch


if __name__ == "__main__":
    retina = RetinaNet(num_classes=3)
    focal_loss = FocalLoss()
    smooth_l1_loss = nn.SmoothL1Loss(reduction='none')
    inp = torch.zeros(1, 3, 256, 256)
    class_out, regr_out = retina(inp)

    boxes = torch.from_numpy(np.array([[10, 20, 120, 220],
                                           [120, 120, 250, 190],
                                           [0, 0, 50, 50]])).view(1, 3, 4)
    labels = torch.from_numpy(np.array([0, 1, 2])).view(1, 3)
    cls_target, regr_target = encode_batch(retina.anchors, boxes, labels)

    # print(focal_loss(class_out, cls_target))

    regr_loss = smooth_l1_loss(regr_out, regr_target) * (cls_target >= 0).float()[..., None]
    cls_loss = focal_loss(class_out, cls_target)
    print(cls_loss.mean(), regr_loss.mean())


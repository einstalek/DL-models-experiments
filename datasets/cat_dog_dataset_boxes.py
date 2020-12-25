import os
import random

import torch

from datasets.utils import parse_xml
import cv2
import numpy as np

from utils.augm import transform_fn, normalize_fn


class CatDogDataset:
    def __init__(self, xml_dir, img_dir, xml_fps, shape=(256, 256),
                 num_classes=2, split='train', return_fp=False):
        super(CatDogDataset, self).__init__()
        self.xml_dir = xml_dir
        self.img_dir = img_dir
        self.xml_fps = xml_fps
        self.shape = shape
        self.num_classes = num_classes
        self.class_idx = {'cat': 0, 'dog': 1}
        self.split = split
        self.return_fp = return_fp

    def _load_sample(self, xml_fp):
        obj = parse_xml(os.path.join(self.xml_dir, xml_fp))
        img_fp = os.path.join(self.img_dir, obj['filename'])
        img = cv2.imread(img_fp)[..., ::-1]
        return img, obj

    def _preprocess_image(self, img, bbox):
        pad_h, pad_w = 0, 0
        h, w, _ = img.shape
        if h > w:
            pad = int((h - w) / 2)
            pad_w = pad
            img = np.pad(img, ((0, 0), (pad, pad), (0, 0)))
        else:
            pad = int((w - h) / 2)
            pad_h = pad
            img = np.pad(img, ((pad, pad), (0, 0), (0, 0)))
        h, w, _ = img.shape
        img = cv2.resize(img, self.shape)
        H, W, _ = img.shape
        scale_x, scale_y = w / W, h / H
        x1, y1, x2, y2 = bbox
        x1, x2 = (x1 + pad_w) / scale_x, (x2 + pad_w) / scale_x
        y1, y2 = (y1 + pad_h) / scale_y, (y2 + pad_h) / scale_y
        return img, (pad_h, pad_w), np.array([x1, y1, x2, y2]) / 256.

    def __getitem__(self, idx):
        if self.split == 'train':
            idx = random.randint(0, len(self) - 1)
        xml_fp = self.xml_fps[idx]
        img, ann = self._load_sample(xml_fp)
        img, pads, bbox = self._preprocess_image(img, ann['bbox'])

        if self.split == 'train':
            args = {'image': img, 'bboxes': [bbox,]}
            augm = transform_fn(**args)
            img, bbox = augm['image'], augm['bboxes'][0]  # box is x1y1x2y2
        x1, y1, x2, y2 = np.array(bbox) * 256
        cx, cy, w, h = (x1+x2)/2, (y1+y2)/2, (x2-x1), (y2-y1)
        img = img / 255.
        img = normalize_fn(img).float()
        box = torch.from_numpy(np.array([cx, cy, w, h])).float()[None]
        labels = np.array([self.class_idx[ann['name']]])
        labels = torch.from_numpy(labels).long()
        return img, box, labels

    def __len__(self):
        return len(self.xml_fps)

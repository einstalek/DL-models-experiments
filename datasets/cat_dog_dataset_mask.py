import os
import random
import numpy as np
import cv2
import torch

from datasets.utils import parse_xml
from utils.augm import transform_fn, normalize_fn


class CatDogDataset:
    def __init__(self, xml_dir, img_dir, xml_fps, shape=(256, 256),
                 num_classes=2, mask_scale=2, split='train',
                 return_fp=False):
        super(CatDogDataset, self).__init__()
        self.xml_dir = xml_dir
        self.img_dir = img_dir
        self.xml_fps = xml_fps
        self.shape = shape
        self.mask_scale = mask_scale
        self.mask_shape = [int(x / self.mask_scale) for x in self.shape]
        self.num_classes = num_classes
        self.class_idx = {'cat': 0, 'dog': 1}
        self.split = split
        self.return_fp = return_fp

    def _load_sample(self, xml_fp):
        obj = parse_xml(os.path.join(self.xml_dir, xml_fp))
        img_fp = os.path.join(self.img_dir, obj['filename'])
        assert os.path.exists(img_fp)
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
        return img, (pad_h, pad_w), (x1, y1, x2, y2)

    def _bbox_to_mask(self, bbox):
        mask = np.zeros(self.mask_shape)
        h, w = self.shape
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(x2, w - 1), min(y2, h - 1)
        if self.mask_scale > 1:
            x1, y1, x2, y2 = [int(t / self.mask_scale) for t in (x1, y1, x2, y2)]
        mask[y1:y2, x1:x2] = 1
        return mask

    def __getitem__(self, idx):
        if self.split == 'train':
            idx = random.randint(0, len(self) - 1)
        xml_fp = self.xml_fps[idx]
        img, ann = self._load_sample(xml_fp)
        img, pads, bbox = self._preprocess_image(img, ann['bbox'])

        mask = np.zeros((self.num_classes, *self.mask_shape))
        _ch = self.class_idx[ann['name']]
        _mask = self._bbox_to_mask(bbox)

        if self.split == 'train':
            args = {'image': img, 'mask': _mask}
            augm = transform_fn(**args)
            img, _mask = augm['image'], augm['mask']
        mask[_ch] += _mask
        img = (img / 255.)
        img = normalize_fn(img).float()
        mask = torch.from_numpy(mask).float()
        if self.return_fp:
            return img, mask, xml_fp
        return img, mask

    def __len__(self):
        return len(self.xml_fps)
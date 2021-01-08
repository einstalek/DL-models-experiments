import os

import albumentations as A
import cv2
import random
import numpy as np

import torch
from torch.utils.data import Dataset

transform_fn = A.Compose(
    [A.HorizontalFlip(),
     A.Rotate(limit=15)],
    additional_targets={'image1': 'image'})


class PairedDataset(Dataset):
    def __init__(self, a_dir, b_dir, split='train', size=(256, 256)):
        self.a_dir = a_dir
        self.b_dir = b_dir
        self.a_fps = os.listdir(a_dir)
        self.a_fps = [os.path.join(a_dir, x) for x in self.a_fps]
        self.b_fps = os.listdir(b_dir)
        self.b_fps = [os.path.join(b_dir, x) for x in self.b_fps]
        self.size = size
        self.split = split

    def __getitem__(self, index):
        if self.split == 'train':
            a_fp = random.sample(self.a_fps, 1)[0]
        else:
            a_fp = self.a_fps[index]
        if self.split == 'train':
            b_fp = f"{a_fp.split('_')[0]}_A.jpg".replace("trainB", "trainA")
        else:
            b_fp = a_fp.replace(f"{self.split}B", f"{self.split}A")

        a_img = self._read(a_fp)
        b_img = self._read(b_fp)

        if self.split == 'train':
            result = transform_fn(image=a_img, image1=b_img)
            a_img = result['image']
            b_img = result['image1']
        a_img = self.get_input(a_img)
        b_img = self.get_input(b_img)
        a_img = torch.from_numpy(a_img).float()
        b_img = torch.from_numpy(b_img).float()
        return a_img, b_img

    def _read(self, fp):
        img = cv2.imread(fp)
        img = cv2.resize(img, self.size)
        return img

    def get_input(self, img):
        img = img / 255
        img = (img - 0.5) / 0.5
        img = np.transpose(img, (2, 0, 1))
        return img

    def __len__(self):
        return len(self.a_fps) * 10
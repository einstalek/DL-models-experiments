import os
import cv2
import random
import json
from collections import defaultdict
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from torchvision import transforms as T


class GarmentParser:
    def __init__(self, root):
        """
        Viton Datset parser
        :param root: full path to the train/test folder with annotations
        """
        self.root = root
        self.image_dir = os.path.join(root, 'image')
        self.image_parse_dir = os.path.join(root, 'image-parse')
        self.cloth_dir = os.path.join(root, 'cloth')
        self.cloth_mask_dir = os.path.join(root, 'cloth-mask')
        self.pose_dir = os.path.join(root, 'pose')
        self.id_map = defaultdict(dict)
        self.build()

    def build(self):
        for img_fp in os.listdir(self.image_dir):
            id = img_fp.split('_')[0]
            self.id_map[id]['image'] = os.path.join(self.image_dir, img_fp)
        for img_mask_fp in os.listdir(self.image_parse_dir):
            id = img_mask_fp.split('_')[0]
            self.id_map[id]['image_mask'] = os.path.join(self.image_parse_dir,
                                                         img_mask_fp)
        for cloth_fp in os.listdir(self.cloth_dir):
            id = cloth_fp.split('_')[0]
            self.id_map[id]['cloth'] = os.path.join(self.cloth_dir, cloth_fp)
        for cloth_mask_fp in os.listdir(self.cloth_mask_dir):
            id = cloth_mask_fp.split('_')[0]
            self.id_map[id]['cloth_mask'] = os.path.join(self.cloth_mask_dir,
                                                         cloth_mask_fp)
        for pose_fp in os.listdir(self.pose_dir):
            id = pose_fp.split('_')[0]
            with open(os.path.join(self.pose_dir, pose_fp)) as f:
                info = json.load(f)
            self.id_map[id]['people'] = info['people']
        self.ids = list(self.id_map.keys())

    def load(self, idx):
        id = self.ids[idx]
        return id, self.id_map[id]

    def load_random(self):
        idx = random.randint(0, len(self.ids) - 1)
        return self.load(idx)

    def __len__(self):
        return len(self.id_map)


def get_normalizer(dim=3, mu=0.5, std=0.5):
    return T.Compose(
        [T.ToTensor(),
         T.Normalize([mu] * dim, [std] * dim)
         ]
    )


transform_fn = A.Compose([
    A.HorizontalFlip(),
    A.ShiftScaleRotate(rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, value=0)],
    keypoint_params=A.KeypointParams(format='xy', remove_invisible=False),
    additional_targets={'image_mask': 'image', 'cloth': 'image', 'cloth_mask': 'image'}
)


class DescriptorDataset(Dataset):
    def __init__(self, gparser: GarmentParser, split='train',
                 image_size=(256, 192), kp_thresh=0.1, kp_rad=10,
                 transform_fn=None):
        """
        Data loader for Viton Dataset as it's described in the original paper
        :param gparser: parser
        :param split: train or val mode
        :param image_size:
        :param kp_thresh: min keypoint confidence to add to the descriptor
        :param kp_rad: size of the positive area in a single key-point mask
        :type transform_fn: augmentation function
        """
        super(DescriptorDataset, self).__init__()
        self.gparser = gparser
        self.split = split
        self.image_size = image_size
        self.kp_thresh = kp_thresh
        self.kp_rad = kp_rad
        self.hid_size = (16, 32)  # size of downsampled body mask
        self.normalizer = {1: get_normalizer(1),
                           3: get_normalizer(3),
                           25: get_normalizer(25)}
        self.transform_fn = transform_fn

    def __getitem__(self, idx):
        """
        :return:
            - descriptor: [kps, body_mask, face, cloth], [25, h, w]
            - target image, [3, h, w]
            - clothing binary mask from target image
            - clothing binary mask from cloth image
        """
        if self.split == 'train':
            id, sample = self.gparser.load_random()
        else:
            id, sample = self.gparser.load(idx)

        image = cv2.imread(sample['image'])[..., ::-1]  # target person image
        image_mask = cv2.imread(sample['image_mask'])[..., ::-1]  # human parsing segmentation map
        cloth = cv2.imread(sample['cloth'])[..., ::-1]  # target clothing item image
        cloth_mask = cv2.imread(sample['cloth_mask'])  # binary mask for clothing item
        keypoints = np.array(sample['people'][0]['pose_keypoints']).reshape(-1, 3)

        # Apply augmentations
        if self.split == 'train' and self.transform_fn is not None:
            augm = transform_fn(image=image, image_mask=image_mask,
                                cloth=cloth, cloth_mask=cloth_mask,
                                keypoints=keypoints[:, :2])
            image = augm['image']
            image_mask = augm['image_mask']
            cloth = augm['cloth']
            cloth_mask = augm['cloth_mask']
            keypoints[:, :2] = augm['keypoints']
        image = image / 255.
        image_mask = image_mask / 255.
        cloth = cloth / 255.
        cloth_mask = cloth_mask.mean(2) / 255.

        # TODO: deal with the hardcoded values for image_mask
        # Select area containing person's hands
        hand_mask = np.where(image_mask[..., 1] > 0.5, 1, 0)
        # Leave only the area around wrists
        hand_mask = hand_mask * self.binary_mask(keypoints, ids=[4, 7], rad=30).sum(2)

        image_mask = image_mask.sum(2)
        # Select area containing person's face and hair
        face_mask = np.where(np.abs(image_mask - 1.) < 0.1, 1, 0)
        # Select area containing lower body clothing items and legs
        lower_body_mask = np.where(np.abs(image_mask - 0.66) < 0.1, 1, 0)
        # Create image containing only face, lower body and hands
        face_pic = image * (hand_mask[..., None] + \
                            face_mask[..., None] + \
                            lower_body_mask[..., None]).astype(np.float32)  #  3 channels
        # Select body area all except the head
        body_mask = np.where(np.abs(image_mask - 1) > 0.1, 1, 0)
        body_mask = (image_mask * body_mask).astype(bool).astype(np.float32)
        # Downsample body mask and then upsample it back
        body_mask_down = cv2.resize(body_mask, self.hid_size)
        body_mask_down = cv2.resize(body_mask_down, self.image_size[::-1])
        body_mask_down = body_mask_down[..., None]  # 1 channel
        # Create masks with key-points
        kps_mask = self.binary_mask(keypoints)  # 18 channels

        desc = np.concatenate([kps_mask,
                               body_mask_down,
                               face_pic,
                               cloth], axis=2)
        upper_cloth_mask = np.where(np.abs(image_mask - 1.3294117) < 0.1, 1, 0)

        return self.preprocess(desc), \
               self.preprocess(image), \
               self.preprocess(upper_cloth_mask), \
               self.preprocess(cloth_mask)

    def binary_mask(self, kps, ids=None, rad=None):
        """
        Generates binary mask around certain key-points
        :param kps: key-points, [N, 3]
        :param ids: ids of key-points
        :param rad: radius of positive area in the mask
        :return: Mask [h, w, N], where N is the number of ids
        """
        if rad is None:
            rad = self.kp_rad
        if ids is not None:
            kps = kps[ids]
        N, _ = kps.shape
        H, W = self.image_size
        mask = np.zeros((*self.image_size, N))
        for i in range(N):
            kp_x, kp_y, kp_proba = kps[i]
            kp_x, kp_y = int(kp_x), int(kp_y)
            if kp_proba < self.kp_thresh:
                continue
            lim_low_x, lim_low_y = max(0, kp_x - rad), max(0, kp_y - rad)
            lim_high_x, lim_high_y = min(W - 1, kp_x + rad), min(H - 1, kp_y + rad)
            for xx in range(lim_low_x, lim_high_x):
                for yy in range(lim_low_y, lim_high_y):
                    if (xx - kp_x) ** 2 + (yy - kp_y) ** 2 < rad ** 2:
                        mask[yy, xx, i] = 1.
        return mask

    def preprocess(self, inp):
        if len(inp.shape) == 2:
            inp = inp[..., None]
        dim = inp.shape[-1]
        inp = self.normalizer[dim](inp.astype(np.float32))
        return inp.float()

    def __len__(self):
        return len(self.gparser)

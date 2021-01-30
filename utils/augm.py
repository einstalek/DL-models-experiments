import torchvision.transforms as T
import albumentations as A


transform_fn = A.Compose([
    A.RandomBrightnessContrast(),
    A.HorizontalFlip(),
    A.GaussNoise(),
    A.ChannelShuffle(),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=60),
])

normalize_fn = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
])
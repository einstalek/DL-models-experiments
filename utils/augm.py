import torchvision.transforms as T
import albumentations as A


transform_fn = A.Compose([
    A.RandomBrightnessContrast(),
    A.HorizontalFlip(),
    A.Rotate(60),
    A.GaussNoise(),
    A.ChannelShuffle(),
])

normalize_fn = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]),
])
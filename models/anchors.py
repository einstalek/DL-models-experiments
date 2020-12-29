import numpy as np
import torch
from torchvision import ops
import matplotlib.pyplot as plt


def plot_bbox(box, mode="xywh"):
    if mode == "xywh":
        x, y, w, h = box
        x1, y1 = x - w/2, y - h/2
        x2, y2 = x + w / 2, y + h / 2
    elif mode == "x1y1x2y2":
        x1, y1, x2, y2 = box
    else:
        raise ValueError
    plt.vlines(x1, y1, y2)
    plt.vlines(x2, y1, y2)
    plt.hlines(y1, x1, x2)
    plt.hlines(y2, x1, x2)


def generate_anchor_boxes(fmap_sizes, subsample, img_size,
                          scales, ratios, mode="xywh", clip=True):
    anchors = []
    for fmap_size in fmap_sizes:
        H = W = fmap_size
        x = np.arange(W) / W
        y = np.arange(H) / H
        yy, xx = np.meshgrid(y, x)
        xy = np.concatenate([xx[..., None], yy[..., None]], -1)
        ct = xy.reshape(-1, 2) * img_size
        N, _ = ct.shape

        for r in ratios:
            for s in scales:
                h = (s / np.sqrt(r)) * img_size * (subsample / fmap_size)
                w = (s * np.sqrt(r)) * img_size * (subsample / fmap_size)
                wh = np.concatenate([np.ones((N, 1)) * h, np.ones((N, 1)) * w], axis=-1)
                if mode == "x1y2x2y2":
                    x1y1 = ct - wh / 2
                    x2y2 = ct + wh / 2
                    temp = np.concatenate([x1y1, x2y2], -1)
                elif mode == "xywh":
                    temp = np.concatenate([ct, wh], -1)
                else:
                    raise ValueError
                anchors.append(temp)
    anchors = np.concatenate(anchors)
    if clip:
        anchors = ops.clip_boxes_to_image(torch.from_numpy(anchors), (img_size, img_size))
        return anchors
    return torch.from_numpy(anchors)


if __name__ == "__main__":
    A = generate_anchor_boxes((32, 16, 8), 8, 256,
                              scales=(0.25,  0.75, 1.25), ratios=(1, 2, 0.5),
                              mode="xywh", clip=False)
    print(A.shape)  # sum(r * s * size**2)
    img = np.zeros((256, 256, 3))
    plt.imshow(img)
    ids = torch.where(torch.logical_and(A[:, 0] == 128, A[:, 1] == 128))
    for box in A[ids]:
        plot_bbox(box, mode="xywh")
    plt.show()

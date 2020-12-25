import matplotlib.pyplot as plt


def plot_bbox(box, c='red', mode="cxcywh"):
    if mode == "cxcywh":
        cx, cy, w, h = map(int, box)
        x1, y1 = cx - w/2, cy - h/2
        x2, y2 = cx + w/2, cy + h/2
    elif mode == "xyxy":
        x1, y1, x2, y2 = map(int, box)
    else:
        raise ValueError
    plt.vlines(x1, y1, y2, color=c)
    plt.vlines(x2, y1, y2, color=c)
    plt.hlines(y1, x1, x2, color=c)
    plt.hlines(y2, x1, x2, color=c)
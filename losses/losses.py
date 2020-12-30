import torch
import torch.nn as nn

class DICEMetrics:
    eps = 1e-7

    def __init__(self, size_average=True, num_classes=2):
        self.size_average = size_average
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        batch_size = outputs.fmap_size(0)
        outputs = outputs.view(batch_size, self.num_classes, -1)
        targets = targets.view(batch_size, self.num_classes, -1)
        nominator = (outputs * targets).sum(dim=2)
        denominator = outputs.sum(dim=2) + targets.sum(dim=2)

        if self.size_average:
            return ((2. * nominator + self.eps) / (denominator + self.eps)).mean()
        return (2. * nominator + self.eps) / (denominator + self.eps)


class BCEDICELoss:
    def __init__(self, loss_weights=None, size_average=True, num_classes=2):
        loss_weights = loss_weights or {'bce': 0.5, 'dice': 0.5}
        self.bce_loss = nn.BCELoss(reduction='elementwise_mean' if size_average else 'none')
        self.dice_loss = DICEMetrics(size_average=size_average, num_classes=num_classes)
        self.loss_weights = loss_weights
        self.size_average = size_average
        self.num_classes = num_classes

    def __call__(self, outputs, targets):
        if self.size_average:
            bce_loss = self.loss_weights['bce'] * self.bce_loss(outputs, targets)
        else:
            bce_loss = (self.loss_weights['bce'] * self.bce_loss(outputs, targets)).view(outputs.fmap_size(0),
                                                                                         -1).mean()
        return bce_loss + self.loss_weights['dice'] * (1 - self.dice_loss(outputs, targets))


class FocalLoss:
    """
    Huge confusion about negative boxes label
    https://stackoverflow.com/questions/53809995/confusions-regarding-retinanet

    Original paper suggests  using alpha=0.25. It makes more sense to use alpha=0.75,
    since positive anchors are minority
    """
    def __init__(self, alpha=0.75, gamma=2, reduction="mean", high_conf_reg=0.):

        self.alpha = alpha
        self.gamma = gamma
        self._loss = nn.CrossEntropyLoss(reduction=reduction)
        self._negative_index = -1
        self._ignore_index = -2
        self.eps = 1e-6
        self.reduction = reduction
        self.high_conf_reg = high_conf_reg

    def __call__(self, outputs, targets):
        """
        :param outputs: (B, A, K), class logits
        :param targets: (B, A, K), {0, 1}
        :return:
            Output is a matrix of logits for each targeet class  [p11, p12, ..., p1K]
        """
        # Filter out ignored anchors
        bsize = outputs.size(0)
        filter_mask = torch.where(targets[..., 0] != self._ignore_index)
        targets = targets[filter_mask]  # (N, K)
        outputs = outputs[filter_mask]  # (N, K)

        # probas = outputs.sigmoid()
        probas = outputs.sigmoid()
        alpha = torch.where(targets != self._negative_index, self.alpha, 1-self.alpha)
        pt = torch.where(targets != self._negative_index, probas, 1-probas)
        pt = pt.clamp(self.eps, 1-self.eps)
        loss = -alpha * (1 - pt)**self.gamma * pt.log() + self.high_conf_reg * outputs.abs().mean()
        if self.reduction == "none":
            return loss / bsize
        elif self.reduction == "sum":
            return loss.sum() / bsize
        elif self.reduction == "mean":
            return loss.mean() / bsize
        else:
            raise ValueError

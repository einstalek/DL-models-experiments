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
    """
    def __init__(self, alpha=0.5, gamma=2):
        self.alpha = alpha
        self.gamma = gamma
        self._loss = nn.CrossEntropyLoss(reduction='none')
        self._ignore_index = -2

    def __call__(self, outputs, targets):
        """
        :param outputs: (B, A, K)
        :param targets: (B, A), from {0, ..., K-1} + {-1, -2}
        :return:

        Output is a matrix of logits for each targeet class  [p11, p12, ..., p1K]
        If the target label L is from {0, ..., K-1}, then loss  would be  -log(1-p1n) - log(p1L)
        If the target is -1 (background), the loss would be -log(1-p1n)
        """
        positive_mask = targets >= 0
        background_mask = targets == -1
        probas = outputs.sigmoid()
        foreground_targets = targets[torch.where(positive_mask)]  # (P,)
        foreground_outputs = outputs[torch.where(positive_mask)]  # (P, K)
        ce_loss = self._loss(foreground_outputs, foreground_targets)
        pt_foreground = torch.gather(foreground_outputs, 1, foreground_targets[:, None]).sigmoid()  # probas of target classes
        pt_background = 1 - probas[background_mask]  # (N, K)
        return torch.mean(self.alpha * ce_loss * (1 - pt_foreground[:, 0]) ** self.gamma) - \
            torch.mean((1 - self.alpha) * (1 - pt_background)**self.gamma * pt_background.log())



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
            bce_loss = (self.loss_weights['bce'] * self.bce_loss(outputs, targets)).view(outputs.fmap_size(0), -1).mean()
        return bce_loss \
               + self.loss_weights['dice'] * (1 - self.dice_loss(outputs, targets))
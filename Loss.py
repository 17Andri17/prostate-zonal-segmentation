import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    """Dice Loss for segmentation"""

    def __init__(self, smooth=1e-6, ignore_background=True):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.ignore_background = ignore_background

    def forward(self, pred, target):
        # pred: [B, C, H, W] probabilities
        # target: [B, C, H, W] one-hot encoded
        # Optionally drop background channel (channel 0)
        if self.ignore_background:
            pred = pred[:, 1:, :, :]
            target = target[:, 1:, :, :]

        intersection = (pred * target).sum(dim=(2, 3))
        union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


class CombinedLoss(nn.Module):
    """Combination of Dice and CrossEntropy losses"""

    def __init__(self, dice_weight=0.5, ce_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.dice_loss = DiceLoss()
        weights = torch.tensor([0.1, 1, 1, 1, 1, 1, 1])
        self.ce_loss = nn.CrossEntropyLoss(weight=weights)
        self.dice_weight = dice_weight
        self.ce_weight = ce_weight

    def forward(self, pred, target):
        if target.dim() == 4:  # One-hot encoded
            # Convert to class indices for CE loss
            target_ce = torch.argmax(target, dim=1)
        else:
            target_ce = target

        dice = self.dice_loss(F.softmax(pred, dim=1), target)
        ce = self.ce_loss(pred, target_ce)

        return self.dice_weight * dice + self.ce_weight * ce

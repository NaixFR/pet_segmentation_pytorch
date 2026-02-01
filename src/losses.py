import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """
    Dice Loss for binary segmentation
    Dice = (2 * intersection + smooth) / (sum + smooth)
    """
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        """
        logits: (B,1,H,W) raw logits
        targets: (B,1,H,W) binary mask {0,1}
        """
        probs = torch.sigmoid(logits)
        probs = probs.view(probs.size(0), -1)
        targets = targets.view(targets.size(0), -1)

        intersection = (probs * targets).sum(dim=1)
        dice = (2.0 * intersection + self.smooth) / (probs.sum(dim=1) + targets.sum(dim=1) + self.smooth)

        return 1.0 - dice.mean()


class BCEDiceLoss(nn.Module):
    """
    BCEWithLogitsLoss + DiceLoss
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss

class FocalLoss(nn.Module):
    """
    Binary Focal Loss (with logits)
    FL = alpha * (1 - pt)^gamma * BCE
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        """
        logits: (B,1,H,W)
        targets: (B,1,H,W) in {0,1}
        """
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)  # pt is prob of correct class

        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce
        return loss.mean()


class FocalDiceLoss(nn.Module):
    """
    Focal Loss + Dice Loss
    """
    def __init__(self, alpha=0.25, gamma=2.0, focal_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.focal = FocalLoss(alpha=alpha, gamma=gamma)
        self.dice = DiceLoss()
        self.focal_weight = focal_weight
        self.dice_weight = dice_weight

    def forward(self, logits, targets):
        focal_loss = self.focal(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.focal_weight * focal_loss + self.dice_weight * dice_loss


if __name__ == "__main__":
    # quick test
    pred_logits = torch.randn(2, 1, 256, 256)
    gt_mask = torch.randint(0, 2, (2, 1, 256, 256)).float()

    criterion = BCEDiceLoss(bce_weight=0.5, dice_weight=0.5)
    loss = criterion(pred_logits, gt_mask)
    print("loss:", loss.item())

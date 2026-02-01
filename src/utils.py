import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
import numpy as np
import os


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)


@torch.no_grad()
def dice_score(preds, targets, eps=1e-7):
    """
    preds: (B,1,H,W) binary {0,1}
    targets: (B,1,H,W) binary {0,1}
    """
    preds = preds.float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = (2 * intersection + eps) / (union + eps)
    return dice.mean().item()


@torch.no_grad()
def iou_score(preds, targets, eps=1e-7):
    """
    IoU = intersection / union
    """
    preds = preds.float()
    targets = targets.float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) - intersection
    iou = (intersection + eps) / (union + eps)
    return iou.mean().item()



@torch.no_grad()
def save_visualization(image, gt_mask, pred_mask, save_path_prefix):
    """
    Save visualization as separate png files:
    - image
    - gt mask
    - pred mask
    - overlay
    save_path_prefix: e.g. outputs/infer_results/result
    """

    img = image.permute(1, 2, 0).cpu().numpy()  # H,W,3
    gt = gt_mask.squeeze(0).cpu().numpy()       # H,W
    pred = pred_mask.squeeze(0).cpu().numpy()   # H,W

    # convert to uint8
    img_u8 = (img * 255).astype(np.uint8)
    gt_u8 = (gt * 255).astype(np.uint8)
    pred_u8 = (pred * 255).astype(np.uint8)

    # overlay: green mask
    overlay = img.copy()
    overlay[..., 1] = np.clip(overlay[..., 1] + pred * 0.6, 0, 1)
    overlay_u8 = (overlay * 255).astype(np.uint8)

    Image.fromarray(img_u8).save(save_path_prefix + "_img.png")
    Image.fromarray(gt_u8).save(save_path_prefix + "_gt.png")
    Image.fromarray(pred_u8).save(save_path_prefix + "_pred.png")
    Image.fromarray(overlay_u8).save(save_path_prefix + "_overlay.png")

    print("Saved:", save_path_prefix + "_img.png")
    print("Saved:", save_path_prefix + "_gt.png")
    print("Saved:", save_path_prefix + "_pred.png")
    print("Saved:", save_path_prefix + "_overlay.png")


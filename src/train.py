import os
import torch
from tqdm import tqdm

from datasets import get_dataloaders
from models import UNet
from src.losses import FocalDiceLoss
from utils import ensure_dir, dice_score, iou_score, save_visualization


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in tqdm(loader, desc="Train", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, masks)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    return running_loss / len(loader)


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device, threshold=0.5):
    model.eval()
    running_loss = 0.0

    dices = []
    ious = []

    for batch in tqdm(loader, desc="Val", leave=False):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        running_loss += loss.item()

        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()

        dices.append(dice_score(preds, masks))
        ious.append(iou_score(preds, masks))

    return running_loss / len(loader), sum(dices) / len(dices), sum(ious) / len(ious)


def main():
    # -------------------------
    # Config (can be moved to yaml later)
    # -------------------------
    # 自动定位项目根目录：pet_segmentation_pytorch/
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    root_dir = os.path.join(project_root, "data")

    img_size = (384, 384)
    batch_size = 4
    lr = 1e-3
    epochs = 10
    num_workers = 2
    base_c = 32  # lightweight UNet

    # paths
    ckpt_dir = "checkpoints"
    out_dir = "outputs"
    ensure_dir(ckpt_dir)
    ensure_dir(out_dir)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # dataloaders
    train_loader, val_loader = get_dataloaders(
        root_dir=root_dir,
        img_size=img_size,
        batch_size=batch_size,
        num_workers=num_workers
    )

    # model
    model = UNet(base_c=base_c).to(device)

    # loss & optimizer
    criterion = FocalDiceLoss(alpha=0.25, gamma=2.0, focal_weight=0.5, dice_weight=0.5)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_dice = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch [{epoch}/{epochs}]")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_dice, val_iou = validate_one_epoch(model, val_loader, criterion, device)

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

        # save best model
        if val_dice > best_dice:
            best_dice = val_dice
            best_path = os.path.join(ckpt_dir, "best.pth")
            torch.save(model.state_dict(), best_path)
            print(f"✅ Best model saved to {best_path} (Dice={best_dice:.4f})")

        # save visualization each epoch (first val sample)
        batch = next(iter(val_loader))
        img = batch["image"][0].to(device)
        gt = batch["mask"][0].to(device)

        model.eval()
        logits = model(img.unsqueeze(0))
        pred = (torch.sigmoid(logits) > 0.5).float()[0]

        vis_path = os.path.join(out_dir, f"epoch_{epoch}.png")
        save_visualization(img.cpu(), gt.cpu(), pred.cpu(), vis_path)
        print(f"Saved visualization: {vis_path}")

    print("\nTraining completed!")
    print(f"Best Val Dice: {best_dice:.4f}")


if __name__ == "__main__":
    main()

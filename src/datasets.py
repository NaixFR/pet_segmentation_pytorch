import os
import random
from typing import Tuple, Dict

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as F


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class PetSegDataset(Dataset):
    """
    Oxford-IIIT Pet Dataset for binary segmentation (pet vs background).
    - images: .jpg
    - masks: trimap .png (values: {1,2,3} -> background, pet, boundary)
    We convert mask to binary: pet=1, background=0 (boundary treated as pet).
    """

    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        img_size: Tuple[int, int] = (256, 256),
        val_ratio: float = 0.2,
        seed: int = 42,
        augment: bool = True
    ):
        """
        Args:
            root_dir: dataset directory, should contain "images" and "annotations".
            split: "train" or "val"
            img_size: resize size (H, W)
            val_ratio: ratio for validation split
            seed: random seed for split
            augment: whether to apply augmentation (only for train)
        """
        assert split in ["train", "val"], "split must be train or val"
        self.root_dir = root_dir
        self.split = split
        self.img_size = img_size
        self.val_ratio = val_ratio
        self.seed = seed
        self.augment = augment

        self.images_dir = os.path.join(root_dir, "images")
        self.masks_dir = os.path.join(root_dir, "annotations", "trimaps")

        if not os.path.isdir(self.images_dir) or not os.path.isdir(self.masks_dir):
            raise FileNotFoundError(
                f"Dataset not found. Expected:\n"
                f"{self.images_dir}\n{self.masks_dir}"
            )

        # list all image filenames
        all_imgs = [f for f in os.listdir(self.images_dir) if f.endswith(".jpg")]
        all_imgs.sort()

        # create train/val split
        set_seed(seed)
        random.shuffle(all_imgs)
        val_size = int(len(all_imgs) * val_ratio)
        val_imgs = all_imgs[:val_size]
        train_imgs = all_imgs[val_size:]

        self.img_files = train_imgs if split == "train" else val_imgs

        print(f"[{split}] dataset size = {len(self.img_files)}")

    def __len__(self):
        return len(self.img_files)

    def _load_image(self, img_path: str) -> Image.Image:
        return Image.open(img_path).convert("RGB")

    def _load_mask(self, mask_path: str) -> Image.Image:
        # mask is single channel png, values in {1,2,3}
        return Image.open(mask_path)

    def _preprocess_mask(self, mask: Image.Image) -> torch.Tensor:
        """
        Convert trimap values to binary mask:
        - background (1) -> 0
        - pet (2) and boundary (3) -> 1
        Output shape: (1, H, W)
        """
        mask_np = np.array(mask).astype(np.int64)

        # original trimap values: 1=background, 2=pet, 3=boundary
        binary_mask = np.where(mask_np == 1, 0, 1).astype(np.float32)

        # to tensor shape (1,H,W)
        binary_mask = torch.from_numpy(binary_mask).unsqueeze(0)
        return binary_mask

    def _augment(self, img: Image.Image, mask: Image.Image):
        """
        Simple augmentation:
        - Random horizontal flip
        - Random color jitter (only image)
        """
        # random horizontal flip
        if random.random() < 0.5:
            img = F.hflip(img)
            mask = F.hflip(mask)

        # random vertical flip (optional, slightly helps)
        if random.random() < 0.2:
            img = F.vflip(img)
            mask = F.vflip(mask)

        # color jitter
        if random.random() < 0.3:
            brightness = 0.2
            contrast = 0.2
            saturation = 0.2
            hue = 0.05
            img = F.adjust_brightness(img, 1 + random.uniform(-brightness, brightness))
            img = F.adjust_contrast(img, 1 + random.uniform(-contrast, contrast))
            img = F.adjust_saturation(img, 1 + random.uniform(-saturation, saturation))
            img = F.adjust_hue(img, random.uniform(-hue, hue))

        return img, mask

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        img_name = self.img_files[idx]
        base_name = img_name.replace(".jpg", "")

        img_path = os.path.join(self.images_dir, img_name)
        mask_path = os.path.join(self.masks_dir, base_name + ".png")

        if not os.path.isfile(mask_path):
            raise FileNotFoundError(f"Mask not found for {img_name}: {mask_path}")

        img = self._load_image(img_path)
        mask = self._load_mask(mask_path)

        # augmentation only for train
        if self.split == "train" and self.augment:
            img, mask = self._augment(img, mask)

        # resize
        img = img.resize(self.img_size[::-1], resample=Image.BILINEAR)
        mask = mask.resize(self.img_size[::-1], resample=Image.NEAREST)

        # to tensor
        img_tensor = F.to_tensor(img)  # (3,H,W), float [0,1]
        mask_tensor = self._preprocess_mask(mask)  # (1,H,W), {0,1}

        return {
            "image": img_tensor,
            "mask": mask_tensor,
            "name": base_name
        }


def get_dataloaders(
    root_dir: str,
    img_size: Tuple[int, int] = (256, 256),
    batch_size: int = 8,
    num_workers: int = 2,
    val_ratio: float = 0.2,
    seed: int = 42,
):
    train_ds = PetSegDataset(
        root_dir=root_dir,
        split="train",
        img_size=img_size,
        val_ratio=val_ratio,
        seed=seed,
        augment=True
    )
    val_ds = PetSegDataset(
        root_dir=root_dir,
        split="val",
        img_size=img_size,
        val_ratio=val_ratio,
        seed=seed,
        augment=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader


if __name__ == "__main__":
    """
    Quick test:
    python src/datasets.py
    """
    if __name__ == "__main__":
        import os

        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        dataset_root = os.path.join(project_root, "data")

        train_loader, val_loader = get_dataloaders(
            root_dir=dataset_root,
            batch_size=4,
            img_size=(256, 256),
            num_workers=0
        )

    batch = next(iter(train_loader))
    print("image shape:", batch["image"].shape)  # (B,3,H,W)
    print("mask shape:", batch["mask"].shape)    # (B,1,H,W)
    print("names:", batch["name"])
    print("mask unique:", torch.unique(batch["mask"]))

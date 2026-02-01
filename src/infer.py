import os
import torch
from PIL import Image
import torchvision.transforms.functional as F

from src.models import UNet
from src.utils import ensure_dir, save_visualization


@torch.no_grad()
def infer_one_image(model, image_path, device, img_size=(384, 384), threshold=0.5):
    # load image
    img = Image.open(image_path).convert("RGB")
    img_resized = img.resize(img_size[::-1])

    img_tensor = F.to_tensor(img_resized).unsqueeze(0).to(device)  # (1,3,H,W)

    # forward
    logits = model(img_tensor)
    prob = torch.sigmoid(logits)
    pred = (prob > threshold).float()[0]  # (1,H,W)

    return F.to_tensor(img_resized), pred


def main():
    # 自动定位项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ckpt_path = os.path.join(project_root, "checkpoints", "best.pth")
    if not os.path.exists(ckpt_path):
        ckpt_path = os.path.join(project_root, "src", "checkpoints", "best.pth")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("best.pth not found in checkpoints/ or src/checkpoints/")

    out_dir = os.path.join(project_root, "outputs", "infer_results")
    ensure_dir(out_dir)

    # device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # model
    model = UNet(base_c=32).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()
    print("Loaded checkpoint:", ckpt_path)

    # choose one test image (你也可以换成你自己的图片路径)
    test_image_path = os.path.join(project_root, "data", "images", "Abyssinian_1.jpg")

    # inference
    img_tensor, pred_mask = infer_one_image(model, test_image_path, device)

    # fake gt mask for visualization (这里我们没有gt，用pred当作展示)
    gt_mask = pred_mask.clone()

    save_prefix = os.path.join(out_dir, "result")
    save_visualization(img_tensor, gt_mask, pred_mask, save_prefix)


if __name__ == "__main__":
    main()

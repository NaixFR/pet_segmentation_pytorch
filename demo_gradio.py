import os
import torch
import gradio as gr
from PIL import Image
import numpy as np
import torchvision.transforms.functional as F

from src.models import UNet


def find_checkpoint(project_root):
    """
    自动兼容两种路径：
    - checkpoints/best.pth
    - src/checkpoints/best.pth
    """
    p1 = os.path.join(project_root, "checkpoints", "best.pth")
    p2 = os.path.join(project_root, "src", "checkpoints", "best.pth")

    if os.path.exists(p1):
        return p1
    elif os.path.exists(p2):
        return p2
    else:
        raise FileNotFoundError("best.pth not found in checkpoints/ or src/checkpoints/")


def overlay_mask_on_image(img_np, mask_np, alpha=0.6):
    """
    img_np: H,W,3 in [0,255]
    mask_np: H,W in {0,1}
    overlay using green channel
    """
    overlay = img_np.copy().astype(np.float32)
    overlay[..., 1] = np.clip(overlay[..., 1] + mask_np * 255 * alpha, 0, 255)
    return overlay.astype(np.uint8)


class SegmentationDemo:
    def __init__(self):
        self.project_root = os.path.dirname(os.path.abspath(__file__))
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        ckpt_path = find_checkpoint(self.project_root)

        self.model = UNet(base_c=32).to(self.device)
        self.model.load_state_dict(torch.load(ckpt_path, map_location=self.device))
        self.model.eval()

        print("✅ Model loaded:", ckpt_path)
        print("✅ Using device:", self.device)

    @torch.no_grad()
    def predict(self, input_image):
        """
        input_image: PIL Image from gradio
        output:
        - mask image
        - overlay image
        """
        if input_image is None:
            return None, None

        # resize
        img = input_image.convert("RGB")
        img_resized = img.resize((384, 384))

        # to tensor
        img_tensor = F.to_tensor(img_resized).unsqueeze(0).to(self.device)

        # forward
        logits = self.model(img_tensor)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred_mask = (probs > 0.5).astype(np.uint8)  # H,W

        # convert to displayable images
        img_np = np.array(img_resized)  # H,W,3 uint8
        mask_vis = (pred_mask * 255).astype(np.uint8)  # H,W
        overlay = overlay_mask_on_image(img_np, pred_mask, alpha=0.6)

        mask_pil = Image.fromarray(mask_vis)
        overlay_pil = Image.fromarray(overlay)

        return mask_pil, overlay_pil


def main():
    demo = SegmentationDemo()

    with gr.Blocks(title="Pet Segmentation Demo (U-Net)") as app:
        gr.Markdown(
            """
            # 🐶🐱 Pet Segmentation Demo (U-Net + PyTorch)
            Upload a pet image → get segmentation mask & overlay result.

            ✅ Model: U-Net (base_c=32)  
            ✅ Dataset: Oxford-IIIT Pet  
            ✅ Output: Binary Mask + Overlay  
            """
        )

        with gr.Row():
            inp = gr.Image(type="pil", label="Input Image")
            out_mask = gr.Image(type="pil", label="Pred Mask")
            out_overlay = gr.Image(type="pil", label="Overlay Result")

        btn = gr.Button("Run Segmentation 🚀")
        btn.click(fn=demo.predict, inputs=inp, outputs=[out_mask, out_overlay])

        # 自动从 data/images 里取 3 张示例图片（保证存在）
        img_dir = os.path.join(demo.project_root, "data", "images")
        example_files = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg")]
        example_files = example_files[:3]

        gr.Examples(
            examples=example_files,
            inputs=inp,
            label="Example Images (click to load)"
        )

    app.launch(share=False)


if __name__ == "__main__":
    main()

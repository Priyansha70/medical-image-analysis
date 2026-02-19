import os
import numpy as np
import torch
from PIL import Image
import cv2

from src.models import build_resnet18
from src.data import get_transforms
from src.explain import GradCAM
from src.utils import device

def main():
    dev = device()

    # load model checkpoint
    model = build_resnet18(num_classes=2, pretrained=False).to(dev)
    ckpt = torch.load("outputs/best.pt", map_location=dev)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # pick one test image (first pneumonia image)
    img_dir = r"data\chest_xray\test\PNEUMONIA"
    imgs = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(("jpg", "jpeg", "png"))])
    if not imgs:
        raise RuntimeError(f"No images found in {img_dir}")

    img_name = imgs[0]
    img_path = os.path.join(img_dir, img_name)

    img = Image.open(img_path).convert("RGB")
    tfm = get_transforms(img_size=224, train=False)
    x = tfm(img).unsqueeze(0).to(dev)

    # gradcam on layer4
    cam = GradCAM(model, model.layer4)
    heatmap, prob = cam(x, class_idx=1)
    cam.close()

    # overlay heatmap
    rgb = np.array(img)
    h, w = rgb.shape[:2]
    hm = cv2.resize(heatmap, (w, h))
    hm_uint8 = np.uint8(255 * hm)

    colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (0.55 * rgb + 0.45 * colored).astype(np.uint8)

    os.makedirs("outputs", exist_ok=True)
    out_path = r"outputs\gradcam_test.png"
    Image.fromarray(overlay).save(out_path)

    print("Saved:", out_path)
    print("PNEUMONIA probability:", round(prob, 4))
    print("Image used:", img_name)

if __name__ == "__main__":
    main()

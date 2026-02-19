import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import streamlit as st
from PIL import Image
import torch
import cv2

from src.models import build_resnet18
from src.data import get_transforms
from src.explain import GradCAM
from src.utils import device


def overlay_heatmap_on_image(rgb_img, heatmap, alpha=0.45):
    h, w = rgb_img.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)

    colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)

    out = (rgb_img * (1 - alpha) + colored * alpha).astype(np.uint8)
    return out


@st.cache_resource
def load_model():
    dev = device()
    model = build_resnet18(num_classes=2, pretrained=False).to(dev)
    ckpt = torch.load("outputs/best.pt", map_location=dev)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, dev


def main():
    st.set_page_config(page_title="Chest X-ray Pneumonia Detector")
    st.title("🩺 Chest X-ray Pneumonia Classifier")
    st.write("Upload an X-ray image to predict NORMAL vs PNEUMONIA with Grad-CAM explanation.")

    uploaded = st.file_uploader("Upload X-ray Image", type=["jpg", "jpeg", "png"])

    if not uploaded:
        return

    img = Image.open(uploaded).convert("RGB")
    st.image(img, caption="Uploaded Image", use_container_width=True)

    model, dev = load_model()

    tfm = get_transforms(img_size=224, train=False)
    x = tfm(img).unsqueeze(0).to(dev)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

    label_map = {0: "NORMAL", 1: "PNEUMONIA"}
    pred_label = label_map[pred_idx]

    st.subheader("Prediction")
    st.write(f"**Prediction:** {pred_label}")
    st.write(f"**Confidence:** {probs[pred_idx]:.3f}")

    st.subheader("Grad-CAM Explanation")

    cam = GradCAM(model, model.layer4)
    heatmap, pneu_prob = cam(x, class_idx=1)
    cam.close()

    overlay = overlay_heatmap_on_image(np.array(img), heatmap)
    st.image(overlay, caption=f"Grad-CAM (PNEUMONIA prob = {pneu_prob:.3f})")

    st.caption("⚠️ For educational/demo purposes only. Not medical advice.")


if __name__ == "__main__":
    main()

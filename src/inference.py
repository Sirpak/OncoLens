from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

from src.data_loader import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from src.model import build_model


def preprocess_image(image: Image.Image, image_size: int) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return transform(image.convert("RGB")).unsqueeze(0)


def load_checkpoint(checkpoint_path: str | Path, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint.get("config", {"model": {"backbone": "resnet18", "pretrained": False}})
    class_names = checkpoint.get("class_names", CLASS_NAMES)
    image_size = int(checkpoint.get("image_size", 224))
    model, target_layer = build_model(config, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device); model.eval()
    return {"model": model, "target_layer": target_layer, "device": device, "class_names": class_names, "image_size": image_size}


def predict_image(bundle: dict, image: Image.Image) -> dict:
    tensor = preprocess_image(image, bundle["image_size"]).to(bundle["device"])
    with torch.no_grad():
        probs = torch.softmax(bundle["model"](tensor), dim=1).squeeze(0).cpu().numpy()
    idx = int(np.argmax(probs))
    return {
        "predicted_class": bundle["class_names"][idx],
        "confidence": float(probs[idx]),
        "probabilities": {name: float(probs[i]) for i, name in enumerate(bundle["class_names"])},
    }


def generate_gradcam(bundle: dict, image: Image.Image) -> Image.Image:
    activations = []; gradients = []
    layer = bundle["target_layer"]
    forward_handle = layer.register_forward_hook(lambda _m, _i, output: activations.append(output.detach()))
    backward_handle = layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0].detach()))
    try:
        tensor = preprocess_image(image, bundle["image_size"]).to(bundle["device"])
        model = bundle["model"]; model.zero_grad(set_to_none=True)
        logits = model(tensor); target_index = int(logits.argmax(dim=1).item())
        logits[:, target_index].backward()
        acts = activations[-1][0]; grads = gradients[-1][0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=0)).cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heatmap = Image.fromarray((cam * 255).astype("uint8")).resize(image.size)
        colored = ImageOps.colorize(heatmap.convert("L"), black="black", white="red")
        return Image.blend(image.convert("RGB"), colored, alpha=0.35)
    finally:
        forward_handle.remove(); backward_handle.remove()
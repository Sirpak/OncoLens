from __future__ import annotations

from torch import nn
from torchvision import models


def build_model(config: dict, num_classes: int):
    model_cfg = config.get("model", {})
    backbone = model_cfg.get("backbone", "resnet18").lower()
    pretrained = bool(model_cfg.get("pretrained", True))

    if backbone == "resnet18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        model = models.resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        target_layer = model.layer4[-1]
    elif backbone == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        target_layer = model.features[-1]
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")

    return model, target_layer
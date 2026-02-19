import torch.nn as nn
from torchvision import models


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    model = models.resnet18(
        weights=models.ResNet18_Weights.DEFAULT if pretrained else None
    )
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        param.requires_grad = False
    for param in model.fc.parameters():
        param.requires_grad = True


def unfreeze_all(model: nn.Module) -> None:
    for p in model.parameters():
        p.requires_grad = True

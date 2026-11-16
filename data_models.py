"""Model constructors and the Food-101-only teacher's inference contract."""
from pathlib import Path

import torch
from torch import nn
from torchvision import models, transforms


def eval_transform():
    return transforms.Compose([
        transforms.Resize(int(224 * 1.14)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


def verify_classes(classes_path, dataset_classes):
    classes = Path(classes_path).read_text(encoding="utf-8").splitlines()
    if len(set(classes)) != 101 or len(classes) != 101:
        raise ValueError(f"Expected 101 distinct class names in {classes_path}")
    if classes != list(dataset_classes):
        raise ValueError("Teacher class order differs from the dataset class order")
    return classes


def load_teacher(checkpoint_path):
    model = models.resnet50(weights=None)
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(model.fc.in_features, 101))
    model.load_state_dict(torch.load(checkpoint_path, weights_only=True, map_location="cpu"))
    return model.requires_grad_(False).eval()


def build_student(pretrained=False):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, 101)
    return model

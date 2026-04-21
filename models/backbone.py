"""Feature extractor backbones for MIL."""

import torch
import torch.nn as nn
from torchvision import models


# Mapping of backbone name -> (builder function, output_dim)
BACKBONE_REGISTRY = {}


def _build_resnet18_feature_extractor(in_channels=1, pretrained=False):
    # Support both new and old torchvision APIs.
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
    except AttributeError:
        backbone = models.resnet18(pretrained=pretrained)

    if in_channels != 3:
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        with torch.no_grad():
            if pretrained:
                if in_channels == 1:
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                elif in_channels > 3:
                    new_conv.weight[:, :3].copy_(old_conv.weight)
                    new_conv.weight[:, 3:].copy_(old_conv.weight[:, :1].repeat(1, in_channels - 3, 1, 1))
                else:
                    new_conv.weight.copy_(old_conv.weight[:, :in_channels])
        backbone.conv1 = new_conv

    # Output shape after this module is Kx512x1x1.
    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    return feature_extractor


def _build_densenet121_feature_extractor(in_channels=1, pretrained=False):
    """DenseNet121 feature extractor. Output: Kx1024x1x1."""
    try:
        weights = models.DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.densenet121(weights=weights)
    except AttributeError:
        backbone = models.densenet121(pretrained=pretrained)

    if in_channels != 3:
        old_conv = backbone.features.conv0
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        with torch.no_grad():
            if pretrained:
                if in_channels == 1:
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                elif in_channels > 3:
                    new_conv.weight[:, :3].copy_(old_conv.weight)
                    new_conv.weight[:, 3:].copy_(old_conv.weight[:, :1].repeat(1, in_channels - 3, 1, 1))
                else:
                    new_conv.weight.copy_(old_conv.weight[:, :in_channels])
        backbone.features.conv0 = new_conv

    feature_extractor = nn.Sequential(
        backbone.features,
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d((1, 1)),
    )
    return feature_extractor


def _build_resnet34_feature_extractor(in_channels=1, pretrained=False):
    """ResNet34 feature extractor. Output: Kx512x1x1."""
    try:
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet34(weights=weights)
    except AttributeError:
        backbone = models.resnet34(pretrained=pretrained)

    if in_channels != 3:
        old_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=False
        )
        with torch.no_grad():
            if pretrained:
                if in_channels == 1:
                    new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
                elif in_channels > 3:
                    new_conv.weight[:, :3].copy_(old_conv.weight)
                    new_conv.weight[:, 3:].copy_(old_conv.weight[:, :1].repeat(1, in_channels - 3, 1, 1))
                else:
                    new_conv.weight.copy_(old_conv.weight[:, :in_channels])
        backbone.conv1 = new_conv

    feature_extractor = nn.Sequential(*list(backbone.children())[:-1])
    return feature_extractor


def build_feature_extractor(backbone_name='resnet18', in_channels=1, pretrained=False):
    """Build a feature extractor by name. Returns (module, output_dim)."""
    name = backbone_name.lower()
    if name == 'resnet18':
        return _build_resnet18_feature_extractor(in_channels, pretrained), 512
    elif name == 'resnet34':
        return _build_resnet34_feature_extractor(in_channels, pretrained), 512
    elif name == 'densenet121':
        return _build_densenet121_feature_extractor(in_channels, pretrained), 1024
    else:
        raise ValueError("Unknown backbone: '{}'. Supported: resnet18, resnet34, densenet121".format(backbone_name))

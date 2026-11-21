"""ResNet-18 (Subject #2) and ResNet-50 (Subject #3).

Datasets: CIFAR-10, ImageNet-1k (val).

Implemented from scratch in pure PyTorch following the original
"Deep Residual Learning for Image Recognition" (He et al., 2015).

Two stem variants are supported:
  * ``stem="imagenet"`` : 7x7 stride-2 conv + 3x3 maxpool (standard ImageNet).
  * ``stem="cifar"``    : 3x3 stride-1 conv, no maxpool (standard CIFAR setup).
"""

from __future__ import annotations

from typing import List, Type, Union

import torch
import torch.nn as nn

__all__ = [
    "ResNet",
    "BasicBlock",
    "Bottleneck",
    "resnet18",
    "resnet50",
]


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    """Residual block used by ResNet-18/34."""

    expansion: int = 1

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module = None
    ) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class Bottleneck(nn.Module):
    """Bottleneck residual block used by ResNet-50/101/152."""

    expansion: int = 4

    def __init__(
        self, inplanes: int, planes: int, stride: int = 1, downsample: nn.Module = None
    ) -> None:
        super().__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        in_channels: int = 3,
        stem: str = "imagenet",
    ) -> None:
        super().__init__()
        if stem not in ("imagenet", "cifar"):
            raise ValueError("stem must be 'imagenet' or 'cifar'")
        self.inplanes = 64

        if stem == "imagenet":
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:  # cifar
            self.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False
            )
            self.maxpool = nn.Identity()
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        downsample = None
        # Project the shortcut when a new stage changes spatial size or channel count.
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


def resnet18(num_classes: int = 1000, stem: str = "imagenet", **kwargs) -> ResNet:
    """ResNet-18. Use ``stem='cifar'`` for CIFAR-10."""
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, stem=stem, **kwargs)


def resnet50(num_classes: int = 1000, stem: str = "imagenet", **kwargs) -> ResNet:
    """ResNet-50 (ImageNet-1k)."""
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, stem=stem, **kwargs)
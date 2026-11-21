"""LeNet-5 / small CNN (Subject #1).

Datasets: MNIST, Fashion-MNIST (28x28 grayscale, 10 classes).

A classic LeNet-5-style network for 28x28 inputs. The layer layout and naming
(``conv1``/``conv2`` + ``fc1``/``fc2``/``fc3``, ReLU activations, 2x2 max-pool,
no input padding so the final feature map is 4x4 -> 16*4*4=256) match the widely
used PyTorch MNIST/adversarial-example LeNet, so public pretrained checkpoints
in that format (e.g. ``ChufanSuki/LeNet5``) load directly. Pure PyTorch.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["LeNet5", "lenet5"]


class LeNet5(nn.Module):
    """Classic LeNet-5 convolutional network.

    Args:
        num_classes: number of output classes (10 for MNIST / Fashion-MNIST).
        in_channels: number of input channels (1 for grayscale).
    """

    def __init__(self, num_classes: int = 10, in_channels: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 6, kernel_size=5)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)  # 28x28 -> 24x24 -> 12x12
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 12x12 -> 8x8 -> 4x4
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def lenet5(num_classes: int = 10, in_channels: int = 1, **kwargs) -> LeNet5:
    """Build a LeNet-5 model."""
    return LeNet5(num_classes=num_classes, in_channels=in_channels, **kwargs)

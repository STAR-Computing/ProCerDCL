"""MobileNetV2 (Subject #4).

Datasets: CIFAR-10, ImageNet-1k (val).

Implemented from scratch in pure PyTorch following
"MobileNetV2: Inverted Residuals and Linear Bottlenecks" (Sandler et al., 2018).

For CIFAR-10 pass ``stem_stride=1`` (and typically a smaller input); for ImageNet
keep the default ``stem_stride=2``.
"""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn

__all__ = ["MobileNetV2", "mobilenet_v2"]


def _make_divisible(v: float, divisor: int = 8, min_value: int = None) -> int:
    """Round channel counts to the nearest multiple of ``divisor``."""
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:  # make sure rounding down does not drop > 10%
        new_v += divisor
    return new_v


class ConvBNReLU(nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int = 3,
        stride: int = 1,
        groups: int = 1,
    ) -> None:
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(
                in_planes,
                out_planes,
                kernel_size,
                stride,
                padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int
    ) -> None:
        super().__init__()
        if stride not in (1, 2):
            raise ValueError("stride must be 1 or 2")
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pointwise expansion
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend(
            [
                # depthwise
                ConvBNReLU(
                    hidden_dim, hidden_dim, stride=stride, groups=hidden_dim
                ),
                # pointwise-linear projection
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            ]
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        num_classes: int = 1000,
        width_mult: float = 1.0,
        in_channels: int = 3,
        stem_stride: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        input_channel = 32
        last_channel = 1280

        # t (expand), c (channels), n (repeats), s (stride)
        inverted_residual_setting = [
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]

        input_channel = _make_divisible(input_channel * width_mult)
        self.last_channel = _make_divisible(last_channel * max(1.0, width_mult))

        features: List[nn.Module] = [
            ConvBNReLU(in_channels, input_channel, stride=stem_stride)
        ]
        for t, c, n, s in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(
                    InvertedResidual(
                        input_channel, output_channel, stride, expand_ratio=t
                    )
                )
                input_channel = output_channel
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*features)

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(self.last_channel, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = nn.functional.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def mobilenet_v2(
    num_classes: int = 1000, stem_stride: int = 2, **kwargs
) -> MobileNetV2:
    """MobileNetV2. Use ``stem_stride=1`` for CIFAR-10."""
    return MobileNetV2(num_classes=num_classes, stem_stride=stem_stride, **kwargs)

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, groups: int = 1) -> None:
        padding = kernel_size // 2
        super().__init__(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True),
        )


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.SiLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = self.shared_mlp(torch.mean(x, dim=(2, 3), keepdim=True))
        mx = self.shared_mlp(torch.amax(x, dim=(2, 3), keepdim=True))
        scale = torch.sigmoid(avg + mx)
        return x * scale


class SpatialAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = torch.mean(x, dim=1, keepdim=True)
        mx = torch.amax(x, dim=1, keepdim=True)
        scale = torch.sigmoid(self.conv(torch.cat([avg, mx], dim=1)))
        return x * scale


class ChannelSpatialAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        self.channel = ChannelAttention(channels, reduction=reduction)
        self.spatial = SpatialAttention()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.spatial(self.channel(x))


class InvertedResidualBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int,
        expansion_ratio: int,
        use_attention: bool,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        hidden_channels = in_channels * expansion_ratio

        self.expand = ConvBNAct(in_channels, hidden_channels, kernel_size=1)
        self.depthwise = ConvBNAct(
            hidden_channels,
            hidden_channels,
            kernel_size=3,
            stride=stride,
            groups=hidden_channels,
        )
        self.attention = (
            ChannelSpatialAttention(hidden_channels)
            if use_attention
            else nn.Identity()
        )
        self.project = nn.Sequential(
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()

        if stride == 1 and in_channels == out_channels:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.expand(x)
        y = self.depthwise(y)
        y = self.attention(y)
        y = self.project(y)
        y = self.dropout(y)
        y = y + self.shortcut(x)
        return self.activation(y)


class FDNetV2(nn.Module):
    """FD-Net V2 architecture matching the manuscript figure (3,068,448 params)."""

    def __init__(self, num_classes: int = 4) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNAct(3, 48, kernel_size=3, stride=2),
            ConvBNAct(48, 48, kernel_size=3, stride=1),
        )

        self.standard_stage = nn.Sequential(
            ConvBNAct(48, 64, kernel_size=3, stride=2),
            ConvBNAct(64, 64, kernel_size=3, stride=1),
        )

        self.stage2 = nn.Sequential(
            InvertedResidualBlock(64, 96, stride=2, expansion_ratio=3, use_attention=False),
            InvertedResidualBlock(96, 96, stride=1, expansion_ratio=3, use_attention=False),
            InvertedResidualBlock(96, 96, stride=1, expansion_ratio=3, use_attention=False),
        )

        self.stage3 = nn.Sequential(
            InvertedResidualBlock(96, 192, stride=2, expansion_ratio=4, use_attention=True, dropout=0.03),
            InvertedResidualBlock(192, 192, stride=1, expansion_ratio=4, use_attention=True, dropout=0.03),
            InvertedResidualBlock(192, 192, stride=1, expansion_ratio=4, use_attention=True, dropout=0.03),
        )

        self.stage4 = nn.Sequential(
            InvertedResidualBlock(192, 320, stride=2, expansion_ratio=4, use_attention=True, dropout=0.05),
            InvertedResidualBlock(320, 320, stride=1, expansion_ratio=4, use_attention=True, dropout=0.05),
        )

        self.final_features = nn.Sequential(
            ConvBNAct(320, 512, kernel_size=1, stride=1),
            ChannelSpatialAttention(512),
        )

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.20),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.standard_stage(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.final_features(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    targets: torch.Tensor,
    temperature: float = 4.0,
    alpha: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if temperature <= 0:
        raise ValueError("temperature must be > 0")
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")

    ce = F.cross_entropy(student_logits, targets)
    kd = F.kl_div(
        F.log_softmax(student_logits / temperature, dim=1),
        F.softmax(teacher_logits / temperature, dim=1),
        reduction="batchmean",
    ) * (temperature ** 2)
    total = alpha * ce + (1.0 - alpha) * kd
    return total, ce, kd


def freeze_module(module: nn.Module) -> nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad = False
    return module

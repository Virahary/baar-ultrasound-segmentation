"""CMUNeXt. Copyright (c) 2023 Fenghe Tang; MIT.

Source revision and modifications: see THIRD_PARTY_NOTICES.md.
"""
from typing import Sequence
import torch
from torch import nn
class Residual(nn.Module):
    def __init__(self, fn: nn.Module) -> None:
        super().__init__()
        self.fn = fn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fn(x) + x


class CMUNeXtBlock(nn.Module):
    def __init__(self, ch_in: int, ch_out: int, depth: int = 1, k: int = 3) -> None:
        super().__init__()
        self.block = nn.Sequential(
            *[
                nn.Sequential(
                    Residual(
                        nn.Sequential(
                            nn.Conv2d(
                                ch_in,
                                ch_in,
                                kernel_size=(k, k),
                                groups=ch_in,
                                padding=(k // 2, k // 2),
                            ),
                            nn.GELU(),
                            nn.BatchNorm2d(ch_in),
                        )
                    ),
                    nn.Conv2d(ch_in, ch_in * 4, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in * 4),
                    nn.Conv2d(ch_in * 4, ch_in, kernel_size=(1, 1)),
                    nn.GELU(),
                    nn.BatchNorm2d(ch_in),
                )
                for _ in range(depth)
            ]
        )
        self.up = conv_block(ch_in, ch_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block(x)
        x = self.up(x)
        return x


class conv_block(nn.Module):
    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                ch_in,
                ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class up_conv(nn.Module):
    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear"),
            nn.Conv2d(
                ch_in,
                ch_out,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        return x


class fusion_conv(nn.Module):
    def __init__(self, ch_in: int, ch_out: int) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                ch_in,
                ch_in,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=2,
                bias=True,
            ),
            nn.GELU(),
            nn.BatchNorm2d(ch_in),
            nn.Conv2d(ch_in, ch_out * 4, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out * 4),
            nn.Conv2d(ch_out * 4, ch_out, kernel_size=(1, 1)),
            nn.GELU(),
            nn.BatchNorm2d(ch_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        return x


class CMUNeXt(nn.Module):
    """Pinned official architecture plus the repository's feature/head API."""

    def __init__(
        self,
        *,
        dims: Sequence[int] = (16, 32, 128, 160, 256),
        depths: Sequence[int] = (1, 1, 1, 3, 1),
        kernels: Sequence[int] = (3, 3, 7, 7, 7),
        num_classes: int = 1,
    ) -> None:
        super().__init__()
        if len(dims) != 5 or len(depths) != 5 or len(kernels) != 5:
            raise ValueError("official CMUNeXt configs require five stages")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive")

        self.dims = tuple(int(value) for value in dims)
        self.depths = tuple(int(value) for value in depths)
        self.kernels = tuple(int(value) for value in kernels)
        self.feature_channels = self.dims[0]

        self.Maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stem = conv_block(ch_in=3, ch_out=self.dims[0])
        self.encoder1 = CMUNeXtBlock(
            ch_in=self.dims[0],
            ch_out=self.dims[0],
            depth=self.depths[0],
            k=self.kernels[0],
        )
        self.encoder2 = CMUNeXtBlock(
            ch_in=self.dims[0],
            ch_out=self.dims[1],
            depth=self.depths[1],
            k=self.kernels[1],
        )
        self.encoder3 = CMUNeXtBlock(
            ch_in=self.dims[1],
            ch_out=self.dims[2],
            depth=self.depths[2],
            k=self.kernels[2],
        )
        self.encoder4 = CMUNeXtBlock(
            ch_in=self.dims[2],
            ch_out=self.dims[3],
            depth=self.depths[3],
            k=self.kernels[3],
        )
        self.encoder5 = CMUNeXtBlock(
            ch_in=self.dims[3],
            ch_out=self.dims[4],
            depth=self.depths[4],
            k=self.kernels[4],
        )

        self.Up5 = up_conv(ch_in=self.dims[4], ch_out=self.dims[3])
        self.Up_conv5 = fusion_conv(
            ch_in=self.dims[3] * 2,
            ch_out=self.dims[3],
        )
        self.Up4 = up_conv(ch_in=self.dims[3], ch_out=self.dims[2])
        self.Up_conv4 = fusion_conv(
            ch_in=self.dims[2] * 2,
            ch_out=self.dims[2],
        )
        self.Up3 = up_conv(ch_in=self.dims[2], ch_out=self.dims[1])
        self.Up_conv3 = fusion_conv(
            ch_in=self.dims[1] * 2,
            ch_out=self.dims[1],
        )
        self.Up2 = up_conv(ch_in=self.dims[1], ch_out=self.dims[0])
        self.Up_conv2 = fusion_conv(
            ch_in=self.dims[0] * 2,
            ch_out=self.dims[0],
        )
        self.Conv_1x1 = nn.Conv2d(
            self.dims[0],
            int(num_classes),
            kernel_size=1,
            stride=1,
            padding=0,
        )

    @property
    def head(self) -> nn.Conv2d:
        """Return the official final projection without duplicate registration."""

        return self.Conv_1x1

    def forward_features(self, image: torch.Tensor) -> torch.Tensor:
        x = image.repeat(1, 3, 1, 1)

        x1 = self.stem(x)
        x1 = self.encoder1(x1)

        x2 = self.Maxpool(x1)
        x2 = self.encoder2(x2)

        x3 = self.Maxpool(x2)
        x3 = self.encoder3(x3)

        x4 = self.Maxpool(x3)
        x4 = self.encoder4(x4)

        x5 = self.Maxpool(x4)
        x5 = self.encoder5(x5)

        d5 = self.Up5(x5)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((x1, d2), dim=1)
        d2 = self.Up_conv2(d2)
        return d2

    def forward(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.forward_features(image)
        logits = self.head(feature)
        return logits, feature

MECHANISM: Cross-stage residual feature preservation

HYPOTHESIS: Adding parameter-free downsampled shortcuts between the successful residual stages will exceed 9,193 correct predictions by preserving earlier-stage features across channel expansion and pooling.

INTENDED_EDIT: Restore the verified residual architecture and all-tensor EMA, then add average-pooled, zero-padded shortcuts around the second and third residual stages without increasing parameter count.

EVIDENCE: Stagewise residual refinement improved validation_correct from 9,172 to 9,193 at the same parameter count, while parameter-only EMA fell to 9,168; this motivates retaining all-tensor EMA and extending the validated feature-preservation mechanism across stage boundaries.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        channels = ((1, 32), (32, 32), (32, 64), (64, 64), (64, 96), (96, 96))
        for index, (in_channels, out_channels) in enumerate(channels):
            layers.extend(
                (
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                    nn.SiLU(inplace=True),
                )
            )
            if index in (1, 3, 5):
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
=======
class ResidualStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.norm2 = nn.BatchNorm2d(out_channels)
        self.pool = nn.MaxPool2d(2)
        self.shortcut_padding = (
            (out_channels - in_channels) // 2 if in_channels > 1 else 0
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm1(self.conv1(images)), inplace=True)
        refinement = self.norm2(self.conv2(features))
        output = self.pool(F.silu(features + refinement, inplace=True))
        if self.shortcut_padding:
            shortcut = F.avg_pool2d(images, kernel_size=2, stride=2)
            shortcut = F.pad(
                shortcut,
                (
                    0,
                    0,
                    0,
                    0,
                    self.shortcut_padding,
                    self.shortcut_padding,
                ),
            )
            output = output + shortcut
        return output


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ResidualStage(1, 32),
            ResidualStage(32, 64),
            ResidualStage(64, 96),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer._ema_tensors = [
        tensor for tensor in model.parameters() if tensor.is_floating_point()
    ]
=======
    optimizer._ema_tensors = [
        tensor
        for tensor in tuple(model.parameters()) + tuple(model.buffers())
        if tensor.is_floating_point()
    ]
>>>>>>> REPLACE
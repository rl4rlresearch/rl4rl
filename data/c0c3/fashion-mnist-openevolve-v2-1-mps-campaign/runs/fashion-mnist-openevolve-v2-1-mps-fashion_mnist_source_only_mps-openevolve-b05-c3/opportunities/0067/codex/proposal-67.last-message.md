MECHANISM: Faster tail-tracking EMA on residual features and normalization statistics

HYPOTHESIS: On the verified residual architecture, increasing all-tensor EMA interpolation from 0.005 to 0.0075 will exceed 9,193 correct predictions by reducing pre-transition influence while retaining the BatchNorm smoothing lost by parameter-only EMA.

INTENDED_EDIT: Restore the validated three-stage residual network and all-tensor EMA, then use a moderately faster 0.0075 EMA rate.

EVIDENCE: Residual stages with all-tensor 0.005 EMA achieved 9,193 correct, versus 9,168 for parameter-only EMA; at 0.0075, less than 1% of the pre-transition average survives the terminal phase, compared with about 4% at 0.005.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm1(self.conv1(images)), inplace=True)
        refinement = self.norm2(self.conv2(features))
        return self.pool(F.silu(features + refinement, inplace=True))


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
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

<<<<<<< SEARCH
                average.lerp_(tensor, 0.005)
=======
                average.lerp_(tensor, 0.0075)
>>>>>>> REPLACE
MECHANISM: Finer curriculum-synchronized EMA tracking

HYPOTHESIS: Increasing the terminal-phase all-tensor EMA rate from 0.01 to 0.015 on the qualified residual design will exceed 9,200 correct predictions by further reducing stale parameter and BatchNorm state after the augmentation transition without abandoning late-training smoothing.

INTENDED_EDIT: Restore the qualified parameter-neutral residual topology and use EMA rate 0.005 during broad augmentation and 0.015 during cardinal augmentation.

EVIDENCE: On the same residual 19/32 curriculum, raising the terminal EMA rate from 0.005 to 0.01 improved validation_correct from 9,193 to 9,200 and cross-entropy from 0.239125 to 0.236378; a smaller continuation tests whether faster inference-aligned tracking remains beneficial.

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

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm1(self.conv1(images)), inplace=True)
        refinement = self.norm2(self.conv2(features))
        return self.pool(F.silu(features + refinement, inplace=True))


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
        else:
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, 0.005)
=======
        else:
            ema_rate = 0.015 if step * 32 >= total_steps * 19 else 0.005
            for average, tensor in zip(
                optimizer._ema_values, optimizer._ema_tensors
            ):
                average.lerp_(tensor, ema_rate)
>>>>>>> REPLACE
MECHANISM: Per-channel residual refinement scaling

HYPOTHESIS: Learnable per-channel gains initialized to reproduce the verified residual network will exceed 9,193 correct predictions by adapting each refinement branch’s contribution without the computational cost of squeeze-excitation.

INTENDED_EDIT: Restore the best 19/32 residual design and add one unit-initialized gain per output channel to each residual branch, adding 192 parameters for a total of 246,010.

EVIDENCE: Stagewise residual refinement improved validation_correct from 9,172 to 9,193 with unchanged parameter count; the more expensive squeeze-excitation extension timed out, motivating an identity-preserving, negligible-overhead test of channel-selective refinement strength.

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
        self.classifier = nn.Sequential(
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
        self.refinement_gain = nn.Parameter(torch.ones(out_channels))
        self.pool = nn.MaxPool2d(2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.norm1(self.conv1(images)), inplace=True)
        refinement = self.norm2(self.conv2(features))
        refinement = refinement * self.refinement_gain.view(1, -1, 1, 1)
        return self.pool(F.silu(features + refinement, inplace=True))


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ResidualStage(1, 32),
            ResidualStage(32, 64),
            ResidualStage(64, 96),
        )
        self.classifier = nn.Sequential(
>>>>>>> REPLACE

<<<<<<< SEARCH
    if step * 64 < total_steps * 39:
=======
    if step * 32 < total_steps * 19:
>>>>>>> REPLACE
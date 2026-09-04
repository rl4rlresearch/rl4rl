MECHANISM: Parameter-neutral residual feature refinement

HYPOTHESIS: Adding a residual path around the second convolution of each stage will exceed 9,167 correct predictions by improving feature reuse and optimization during the fixed 100,000-example exposure.

INTENDED_EDIT: Preserve the validated 39/64 augmentation curriculum, spatial classifier, optimizer, EMA, loss, and test-time ensemble while converting each convolution pair into a residual stage with unchanged learned-parameter count.

EVIDENCE: Spatial attention pooling reduced validation correct to 9,103, indicating that the position-sensitive head should be retained; this patch instead improves the feature extractor’s optimization topology without consuming the remaining parameter budget or adding substantial computation.

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
        self.expand = nn.Sequential(
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
        self.refine = nn.Sequential(
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        self.activation = nn.SiLU(inplace=True)
        self.pool = nn.MaxPool2d(2)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        features = self.expand(images)
        features = self.activation(features + self.refine(features))
        return self.pool(features)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            ResidualStage(1, 32),
            ResidualStage(32, 64),
            ResidualStage(64, 96),
        )
>>>>>>> REPLACE
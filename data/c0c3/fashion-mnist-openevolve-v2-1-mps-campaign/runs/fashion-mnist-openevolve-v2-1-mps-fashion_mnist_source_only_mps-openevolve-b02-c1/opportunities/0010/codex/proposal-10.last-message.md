MECHANISM: Residual learned-downsampling pyramid with global mean–max pooling

HYPOTHESIS: Replacing destructive max pooling and the large position-specific flattening head with residual learned downsampling and global mean–max feature aggregation will exceed 9,238 correct validation predictions.

INTENDED_EDIT: Replace the sequential CNN with a seven-convolution residual pyramid that preserves information through projection shortcuts and classifies pooled channel statistics; retain the proven augmentation, optimizer, batch size, schedule, and test-time ensemble.

EVIDENCE: The current design reached 9,238 correct, while another batch-size reduction timed out, limiting further gains from additional optimizer steps. Roughly half of its parameters are committed to the 3×3 flattening head, so reallocating capacity to residual spatial feature extraction tests a genuinely different representation within the parameter and runtime constraints.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 128),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(128, 10),
        )
=======
class ResidualStage(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
        )
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.main(features) + self.skip(features))


class GlobalMeanMaxPool(nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        mean_features = features.mean(dim=(-2, -1))
        max_features = features.amax(dim=(-2, -1))
        return torch.cat((mean_features, max_features), dim=1)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 28, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(28),
            nn.GELU(),
            ResidualStage(28, 28),
            ResidualStage(28, 56, stride=2),
            ResidualStage(56, 88, stride=2),
        )
        self.classifier = nn.Sequential(
            GlobalMeanMaxPool(),
            nn.Linear(88 * 2, 256),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(256, 10),
        )
>>>>>>> REPLACE
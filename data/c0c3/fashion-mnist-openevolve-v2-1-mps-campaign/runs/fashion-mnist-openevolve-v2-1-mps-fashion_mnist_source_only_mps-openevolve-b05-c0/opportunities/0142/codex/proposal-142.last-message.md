MECHANISM: Projected residual convolutional block

HYPOTHESIS: Adding a learned residual path across the two 14×14 convolutional layers will exceed 9,260 correct predictions by improving feature and gradient propagation within the fixed two-pass training budget.

INTENDED_EDIT: Replace the two later sequential convolutions with an equivalent residual block using a 1×1 BatchNorm projection, while preserving all training, EMA, calibration, and batch-size settings.

EVIDENCE: Both directions of label-smoothing adjustment and both tested input augmentations reduced validation correctness; this motivates an orthogonal architecture-level optimization that preserves spatial semantics and remains below the parameter ceiling.

<<<<<<< SEARCH
BATCH_SIZE = 32
GRAD_CLIP_NORM = 1.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 32
GRAD_CLIP_NORM = 1.0


class ResidualFeatureBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
        )
        self.skip = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=1),
            nn.BatchNorm2d(48),
        )
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(self.main(features) + self.skip(features))


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            ResidualFeatureBlock(),
            nn.MaxPool2d(2),
        )
>>>>>>> REPLACE
MECHANISM: Parameter-efficient residual spatial refinement

HYPOTHESIS: Adding depthwise-separable residual refinement at every resolution will exceed 9,210 correct predictions by learning deeper local shape transformations without sacrificing the successful spatial classifier or exceeding 250,000 parameters.

INTENDED_EDIT: Insert four residual refinement blocks after the existing resolution stages and reduce the classifier width from 160 to 136, yielding approximately 249,189 learned parameters.

EVIDENCE: Global-context and axial-profile alternatives fell to 9,128 and 9,167 correct, while the spatial flattened design reached 9,210; this suggests spatial organization is load-bearing, but challenges the shared assumption that one convolution per resolution provides sufficient learned refinement.

<<<<<<< SEARCH
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0


class ImageClassifier(nn.Module):
=======
BATCH_SIZE = 64
GRAD_CLIP_NORM = 5.0


class ResidualRefinement(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=3,
            padding=1,
            groups=channels,
            bias=False,
        )
        self.depthwise_norm = nn.BatchNorm2d(channels)
        self.pointwise = nn.Conv2d(
            channels,
            channels,
            kernel_size=1,
            bias=False,
        )
        self.pointwise_norm = nn.BatchNorm2d(channels)
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        residual = self.depthwise(features)
        residual = self.activation(self.depthwise_norm(residual))
        residual = self.pointwise_norm(self.pointwise(residual))
        return self.activation(features + residual)


class ImageClassifier(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
=======
        self.features = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            ResidualRefinement(24),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
            ResidualRefinement(48),
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.GELU(),
            nn.MaxPool2d(2),
            ResidualRefinement(72),
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            ResidualRefinement(96),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, 10),
=======
            nn.Linear(96 * 3 * 3, 136),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(136, 10),
>>>>>>> REPLACE
MECHANISM: Compute-neutral residual feature reallocation

HYPOTHESIS: Reallocating parameters from the oversized dense head into wider multiscale convolutional features and a residual depthwise block will raise validation_correct above 9,260 without the runtime cost of test-time ensembling.

INTENDED_EDIT: Replace the three-convolution feature extractor with a similarly efficient pooled residual CNN, retaining the optimizer, loss, EMA, batch size, and verified evaluation calibration; the new model has 249,767 learned parameters.

EVIDENCE: Evaluation calibration has saturated at 9,260 correct, while reflection reduced accuracy and translation-based training or ensembling either reduced accuracy or timed out, motivating a compute-conscious architectural improvement without augmentation or extra evaluation forwards.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(48 * 7 * 7, 88),
            nn.GELU(),
            nn.Linear(88, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = logits * 1.416375
        return logits
=======
class ResidualDepthwiseBlock(nn.Module):
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
            channels, channels, kernel_size=1, bias=False
        )
        self.pointwise_norm = nn.BatchNorm2d(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        features = self.depthwise(inputs)
        features = F.gelu(self.depthwise_norm(features))
        features = self.pointwise_norm(self.pointwise(features))
        return F.gelu(inputs + features)


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(
                1, 32, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                32, 56, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(56),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(
                56, 96, kernel_size=3, padding=1, bias=False
            ),
            nn.BatchNorm2d(96),
            nn.GELU(),
            ResidualDepthwiseBlock(96),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 3 * 3, 199),
            nn.GELU(),
            nn.Linear(199, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.classifier(self.features(images))
        if not self.training:
            logits = logits * 1.416375
        return logits
>>>>>>> REPLACE
MECHANISM: Residual learned-downsampling feature hierarchy

HYPOTHESIS: Replacing fixed max pooling and the oversized dense head with residual strided-convolution downsampling and a third convolutional stage will exceed 9,282 correct predictions by learning which spatial information to preserve at each resolution.

INTENDED_EDIT: Challenge the assumption that fixed max pooling plus a high-dimensional flattened head is the best capacity allocation; use learned residual downsampling, progressive 32→48→72 channels, and a compact classifier while retaining the proven paired-view training and EMA.

EVIDENCE: Simply deepening the existing max-pooled branches regressed from 9,212 to 9,196, and translation ensembling fell to 9,187, suggesting that neither more same-resolution convolutions nor inference-time correction solves the representation bottleneck. The 9,282-result paired objective remains intact while this patch tests a different spatial computation within the parameter ceiling.

<<<<<<< SEARCH
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.residual1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.transition = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
        )
        self.pool = nn.MaxPool2d(2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 48),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(48, 10),
        )

    def _forward_view(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.residual1(features))
        features = self.pool(features)
        features = self.transition(features)
        features = F.gelu(features + self.residual2(features))
        features = self.pool(features)
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_view(images)
        if self.training:
            return logits
        flipped_logits = self._forward_view(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
=======
class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
        )
        self.residual1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(
                32, 48, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(48),
        )
        self.down1_skip = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(
                48, 72, kernel_size=3, stride=2, padding=1, bias=False
            ),
            nn.BatchNorm2d(72),
        )
        self.down2_skip = nn.Sequential(
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(48, 72, kernel_size=1, bias=False),
            nn.BatchNorm2d(72),
        )
        self.residual3 = nn.Sequential(
            nn.Conv2d(72, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(72 * 7 * 7, 32),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(32, 10),
        )

    def _forward_view(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(features + self.residual1(features))
        features = F.gelu(
            self.down1(features) + self.down1_skip(features)
        )
        features = F.gelu(features + self.residual2(features))
        features = F.gelu(
            self.down2(features) + self.down2_skip(features)
        )
        features = F.gelu(features + self.residual3(features))
        return self.classifier(features)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits = self._forward_view(images)
        if self.training:
            return logits
        flipped_logits = self._forward_view(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
>>>>>>> REPLACE
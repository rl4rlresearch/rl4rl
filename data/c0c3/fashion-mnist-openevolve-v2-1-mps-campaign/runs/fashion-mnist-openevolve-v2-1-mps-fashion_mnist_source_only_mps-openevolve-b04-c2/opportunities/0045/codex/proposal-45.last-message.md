MECHANISM: Accuracy-boundary batch-normalization refinement

HYPOTHESIS: Momentum 0.006225 will retain 9,279 correct predictions while lowering cross-entropy below 0.208620245.

INTENDED_EDIT: Set all four batch-normalization momenta from 0.0062 to 0.006225, preserving every other setting.

EVIDENCE: Momentum 0.0062 retained 9,279 correct, whereas 0.00625 lowered cross-entropy but lost one prediction; their midpoint most directly probes the narrow accuracy boundary.

<<<<<<< SEARCH
        self.norm = nn.BatchNorm2d(128, momentum=0.0062)
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(features)))


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.0062),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.0062),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.0062),
=======
        self.norm = nn.BatchNorm2d(128, momentum=0.006225)
        self.activation = nn.GELU()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.activation(self.norm(self.conv(features)))


class ImageClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=0.006225),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, momentum=0.006225),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128, momentum=0.006225),
>>>>>>> REPLACE
MECHANISM: Parameter-free intra-stage residual connections

HYPOTHESIS: Adding identity skips around the second convolution of each stage will exceed 9,290 correct predictions by improving feature preservation and gradient flow while retaining the verified 247,836-parameter head, augmentation, optimizer, and ensemble.

INTENDED_EDIT: Replace the sequential convolutional extractor with equivalent explicitly named layers and add residual additions within both constant-width convolutional stages.

EVIDENCE: Expanding the existing dense bottleneck improved validation correct from 9,265 to 9,290, whereas replacing head capacity with a fifth convolution regressed; this motivates preserving all successful capacity while testing parameter-free connectivity in the existing extractor.

<<<<<<< SEARCH
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.SiLU(),
            nn.MaxPool2d(2),
        )
=======
        self.conv1 = nn.Conv2d(
            1, 32, kernel_size=3, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(
            32, 32, kernel_size=3, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(
            32, 64, kernel_size=3, padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(
            64, 64, kernel_size=3, padding=1, bias=False
        )
        self.bn4 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = F.silu(self.bn1(self.conv1(images)))
        features = F.silu(self.bn2(self.conv2(features)) + features)
        features = self.pool(features)

        features = F.silu(self.bn3(self.conv3(features)))
        features = F.silu(self.bn4(self.conv4(features)) + features)
        features = self.pool(features)
        return self.classifier(features)
>>>>>>> REPLACE
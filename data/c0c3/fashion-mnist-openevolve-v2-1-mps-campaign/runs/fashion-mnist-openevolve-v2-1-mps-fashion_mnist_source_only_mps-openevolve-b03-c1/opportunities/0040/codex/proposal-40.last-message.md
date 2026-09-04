MECHANISM: Normalized multi-scale hypercolumn fusion

HYPOTHESIS: Feeding coarse stage-2 and stage-3 feature maps directly into the positional classifier will exceed 9,133 correct predictions by preserving mid-level shape and texture evidence that the current deepest-only representation discards.

INTENDED_EDIT: Split the convolutional hierarchy into stages, fuse normalized 12-channel projections from its intermediate scales with the final 3×3 map, and resize the dense head; the model remains lightweight at 249,443 learned parameters.

EVIDENCE: Translation augmentation and multi-view inference produced repeated gains, indicating that representations stable across nearby spatial scales are useful. Unlike the timed-out widened residual/global-pooling design, this tests multi-scale prediction with negligible added convolutional work while retaining the successful positional final-grid readout.

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
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(96 * 3 * 3, 160),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(160, 10),
        )
=======
        self.stage1 = nn.Sequential(
            nn.Conv2d(4, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.stage2 = nn.Sequential(
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.stage3 = nn.Sequential(
            nn.Conv2d(48, 72, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(72),
            nn.GELU(),
            nn.MaxPool2d(2),
        )
        self.stage4 = nn.Sequential(
            nn.Conv2d(72, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
        )
        self.fuse2 = nn.Sequential(
            nn.Conv2d(48, 12, kernel_size=1, bias=False),
            nn.BatchNorm2d(12),
            nn.GELU(),
        )
        self.fuse3 = nn.Sequential(
            nn.Conv2d(72, 12, kernel_size=1, bias=False),
            nn.BatchNorm2d(12),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.15),
            nn.Linear(120 * 3 * 3, 131),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(131, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        details = F.conv2d(padded, self.detail_kernels)
        represented = torch.cat((images, details), dim=1)
        return self.classifier(self.features(represented))
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        padded = F.pad(images, (1, 1, 1, 1), mode="replicate")
        details = F.conv2d(padded, self.detail_kernels)
        represented = torch.cat((images, details), dim=1)

        level1 = self.stage1(represented)
        level2 = self.stage2(level1)
        level3 = self.stage3(level2)
        deep = self.stage4(level3)

        coarse = F.adaptive_avg_pool2d(self.fuse2(level2), (3, 3))
        middle = self.fuse3(level3)
        fused = torch.cat((deep, middle, coarse), dim=1)
        return self.classifier(fused)
>>>>>>> REPLACE
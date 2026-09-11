MECHANISM: Deep low-resolution spatial-pyramid evidence network

HYPOTHESIS: Reallocating the oversized flatten bottleneck into six full-context residual blocks and a multi-scale pooling head will exceed 9,251 correct predictions while remaining within the runtime and parameter limits.

INTENDED_EDIT: Downsample to a 152-channel 4×4 representation, perform six residual depthwise context updates, and classify concatenated 2×2 average-pooled and global-max evidence; training and EMA evaluation remain unchanged.

EVIDENCE: The current design spends roughly 73% of its 245,040 parameters on one 4,704→38 bottleneck and has plateaued at 9,251 correct despite extensive ensemble tuning. Prior high-resolution additions timed out, so this patch performs the added representation learning at 4×4 resolution and uses approximately 239,234 parameters with only a modest FLOP increase.

<<<<<<< SEARCH
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            ResidualDepthwiseBlock(48),
            nn.MaxPool2d(2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(96),
            nn.GELU(),
            ResidualDepthwiseBlock(96),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * 7 * 7, 38),
            nn.LayerNorm(38),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(38, 10),
        )
=======
        self.stem = nn.Sequential(
            nn.Conv2d(1, 24, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(24),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(24, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            ResidualDepthwiseBlock(48),
            nn.MaxPool2d(2),
            nn.Conv2d(
                48,
                152,
                kernel_size=3,
                stride=2,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(152),
            nn.GELU(),
            *[ResidualDepthwiseBlock(152) for _ in range(6)],
        )
        self.avg_pool = nn.AdaptiveAvgPool2d((2, 2))
        self.max_pool = nn.AdaptiveMaxPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.LayerNorm(152 * 5),
            nn.GELU(),
            nn.Dropout(p=0.1),
            nn.Linear(152 * 5, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.stem(images))
=======
    def _classify(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        pooled = torch.cat(
            (
                self.avg_pool(features).flatten(1),
                self.max_pool(features).flatten(1),
            ),
            dim=1,
        )
        return self.classifier(pooled)
>>>>>>> REPLACE
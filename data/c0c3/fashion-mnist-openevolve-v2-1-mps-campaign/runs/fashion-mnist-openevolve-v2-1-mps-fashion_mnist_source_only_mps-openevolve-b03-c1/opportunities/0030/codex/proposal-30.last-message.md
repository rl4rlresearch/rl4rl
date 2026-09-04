MECHANISM: Late residual semantic refinement with dual-statistic invariant readout

HYPOTHESIS: Replacing the fixed 3×3 flattening head with mean-and-maximum spatial summaries, while spending the recovered capacity on a late residual bottleneck, will exceed 9,133 correct predictions by learning stronger whole-object features that are less sensitive to crop position.

INTENDED_EDIT: Preserve the successful differential inputs and early hierarchy, widen only the final 3×3 representation to 128 channels, add a low-cost residual refinement block, and classify normalized global mean/max features; the resulting model has 245,258 learned parameters.

EVIDENCE: Translation augmentation and multi-view inference produced repeated gains, challenging the assumption that absolute final-grid coordinates are essential. The earlier attention/peak-pooling design timed out after also widening the hierarchy and preserving a 4×4 map; this patch isolates spatially invariant prediction with a smaller 3×3 bottleneck and no attention or depthwise operations.

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
            nn.Conv2d(72, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        self.refine = nn.Sequential(
            nn.Conv2d(128, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(48),
            nn.GELU(),
            nn.Conv2d(48, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(0.15),
            nn.Linear(256, 320),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(320, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        represented = torch.cat((images, details), dim=1)
        return self.classifier(self.features(represented))
=======
        represented = torch.cat((images, details), dim=1)
        feature_map = self.features(represented)
        feature_map = F.gelu(feature_map + self.refine(feature_map))
        pooled = torch.cat(
            (
                feature_map.mean(dim=(-2, -1)),
                feature_map.amax(dim=(-2, -1)),
            ),
            dim=1,
        )
        return self.classifier(pooled)
>>>>>>> REPLACE
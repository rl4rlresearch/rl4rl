MECHANISM: Bounded content-conditioned feature canonicalization

HYPOTHESIS: A learned affine warp of the 7×7 feature grid will exceed 9,267 correct predictions by correcting image-specific pose and translation before classification, while retaining the spatial layout that pooling-based designs lost.

INTENDED_EDIT: Replace the fixed-grid assumption with an identity-initialized, bounded spatial transformer over stem features; this adds 11,766 parameters for a total of 245,200.

EVIDENCE: Uniform crop changes and forced rotation augmentation regressed to 9,250 and 9,203, while position-free pooling also regressed, indicating that indiscriminate invariance is harmful and spatial layout is load-bearing. A learned per-image canonicalization can selectively correct nuisance geometry without discarding that layout.

<<<<<<< SEARCH
        self.residual = nn.Sequential(
            nn.Conv2d(40, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(56),
            nn.GELU(),
            nn.Conv2d(56, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(56),
        )
=======
        self.feature_aligner = nn.Linear(40 * 7 * 7, 6)
        nn.init.zeros_(self.feature_aligner.weight)
        nn.init.zeros_(self.feature_aligner.bias)
        self.residual = nn.Sequential(
            nn.Conv2d(40, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(56),
            nn.GELU(),
            nn.Conv2d(56, 56, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(56),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.stem(images)
        affine_delta = torch.tanh(
            self.feature_aligner(features.flatten(1))
        ).view(-1, 2, 3)
        identity = features.new_tensor(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0))
        ).unsqueeze(0)
        transform_limits = features.new_tensor(
            ((0.06, 0.08, 0.16), (0.08, 0.06, 0.16))
        ).unsqueeze(0)
        theta = identity + transform_limits * affine_delta
        sampling_grid = F.affine_grid(
            theta,
            features.shape,
            align_corners=False,
        )
        features = F.grid_sample(
            features,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        features = F.gelu(self.residual(features) + self.shortcut(features))
        return self.classifier(features)
>>>>>>> REPLACE
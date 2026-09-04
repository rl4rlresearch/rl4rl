MECHANISM: Paired-view batch-stat normalization

HYPOTHESIS: Evaluating both mirrored views together with live BatchNorm statistics will exceed 9,328 correct predictions by eliminating the mismatch between EMA parameters and non-averaged running statistics.

INTENDED_EDIT: Disable BatchNorm running statistics and process paired views jointly during both training and evaluation, preserving training computation and parameter count.

EVIDENCE: The 9,328-correct design copies final BatchNorm buffers onto EMA parameters, while prior attempts to improve normalization-state alignment timed out; this tests that unresolved mechanism without adding learned parameters or training compute.

<<<<<<< SEARCH
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
=======
        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, track_running_stats=False),
            nn.GELU(),
        )
        self.residual1 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32, track_running_stats=False),
        )
        self.transition = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=False),
            nn.GELU(),
        )
        self.residual2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64, track_running_stats=False),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.training:
            batch_size = images.shape[0]
            paired_images = torch.cat(
                (images, torch.flip(images, dims=(-1,))),
                dim=0,
            )
            paired_features = self._forward_features(paired_images)
            features = paired_features[:batch_size]
            flipped_features = paired_features[batch_size:]
        else:
            features = self._forward_features(images)
            flipped_features = self._forward_features(
                torch.flip(images, dims=(-1,))
            )

        logits = self._classify_views(features, flipped_features)
=======
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        batch_size = images.shape[0]
        paired_images = torch.cat(
            (images, torch.flip(images, dims=(-1,))),
            dim=0,
        )
        paired_features = self._forward_features(paired_images)
        features = paired_features[:batch_size]
        flipped_features = paired_features[batch_size:]

        logits = self._classify_views(features, flipped_features)
>>>>>>> REPLACE
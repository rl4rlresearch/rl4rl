MECHANISM: Dual-statistic spatial pyramid classifier

HYPOTHESIS: Replacing exact-grid flattening with multiscale regional mean/max pooling and a wider nonlinear bottleneck will exceed 9,240 correct predictions by learning shape evidence that is less sensitive to residual translations.

INTENDED_EDIT: Challenge the assumption that every 7×7 location needs an independent dense weight: summarize features over 1×1, 2×2, and 4×4 regions using average and maximum evidence, expand the bottleneck from 48 to 56 units, and retain the best verified smoothing and calibration.

EVIDENCE: The successful design relies heavily on translation augmentation and 25-shift evaluation, indicating exact feature alignment is a nuisance. Unlike the failed multiplicative coordinate gating, this pyramid preserves regional layout and feature content; unlike self-attention, it adds negligible computational cost. Reference Design 1 establishes 0.04→0 smoothing as the strongest verified training objective.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(80 * 7 * 7, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )
=======
        self.classifier = nn.Sequential(
            nn.Linear(80 * 2 * (1 + 4 + 16), 56),
            nn.LayerNorm(56),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(56, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
=======
    @staticmethod
    def _spatial_pyramid(features: torch.Tensor) -> torch.Tensor:
        summaries = []
        for size in (1, 2, 4):
            summaries.append(
                F.adaptive_avg_pool2d(features, size).flatten(1)
            )
            summaries.append(
                F.adaptive_max_pool2d(features, size).flatten(1)
            )
        return torch.cat(summaries, dim=1)

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        return self.classifier(self._spatial_pyramid(features))
>>>>>>> REPLACE

<<<<<<< SEARCH
        return 1.30 * logits
=======
        return 1.296352 * logits
>>>>>>> REPLACE

<<<<<<< SEARCH
    smoothing = 0.03 + 0.02 * math.cos(math.pi * progress)
=======
    smoothing = 0.02 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE
MECHANISM: Residual global-moment evidence pooling

HYPOTHESIS: Adding a translation-tolerant prediction branch over channel-wise mean and maximum activations will exceed 9,240 correct predictions by complementing the existing absolute-position classifier with global shape and feature-presence evidence, while retaining the proven local representation.

INTENDED_EDIT: Preserve the strongest backbone, training curricula, and spatial classifier; add a 5,802-parameter global-moment head, blend its logits residually, and retain the proven 1.10 evaluation calibration. Total learned parameters become 248,348.

EVIDENCE: Optimization and loss-schedule refinements have plateaued at 9,240 correct, while the strong reliance on translation augmentation and 25-view evaluation indicates that the flattened 7×7 classifier’s position sensitivity is load-bearing. The unverified multiscale replacement discarded that strong route entirely; this patch cleanly tests translation-tolerant pooled evidence as a complementary learned mechanism. Reference Design 1 also shows that 1.10 logit scaling preserves accuracy while lowering cross-entropy.

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
            nn.Flatten(),
            nn.Linear(80 * 7 * 7, 48),
            nn.LayerNorm(48),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(48, 10),
        )
        self.global_classifier = nn.Sequential(
            nn.LayerNorm(160),
            nn.Linear(160, 32),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(32, 10),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        return self.classifier(features)
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        features = self.features(self._image_basis(images))
        features = F.gelu(features + self.refine(features))
        global_statistics = torch.cat(
            (
                features.mean(dim=(-2, -1)),
                features.amax(dim=(-2, -1)),
            ),
            dim=1,
        )
        return (
            self.classifier(features)
            + 0.35 * self.global_classifier(global_statistics)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        return logits
=======
        return 1.10 * logits
>>>>>>> REPLACE
MECHANISM: Parallel spatial-statistics residual head

HYPOTHESIS: Adding a normalized class-specific residual path from each channel’s spatial mean, standard deviation, and maximum will exceed 9,314 correct predictions by capturing translation-tolerant texture and feature-presence evidence that the shared 30-dimensional flattened bottleneck may discard.

INTENDED_EDIT: Preserve the successful fixed-position classifier, but challenge the assumption that all predictions must pass through its 30 shared features by adding a zero-initialized global-statistics head; total learned parameters remain below 250,000.

EVIDENCE: Bottleneck BatchNorm improved correctness from 9,249 to 9,314, while replacing fixed spatial structure with content-addressed pooling fell to 9,228. This motivates a normalized complementary path that retains the proven classifier rather than replacing it.

<<<<<<< SEARCH
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 30),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(30, 10),
        )

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(images))
=======
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 30),
            nn.BatchNorm1d(30),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(30, 10),
        )
        self.statistics_norm = nn.BatchNorm1d(64 * 3)
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)

    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(images)
        spatial_mean = feature_map.mean(dim=(2, 3))
        spatial_std = feature_map.var(
            dim=(2, 3), unbiased=False
        ).add(1e-4).sqrt()
        spatial_max = feature_map.amax(dim=(2, 3))
        statistics = torch.cat(
            (spatial_mean, spatial_std, spatial_max), dim=1
        )
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        return self.classifier(feature_map) + residual_logits
>>>>>>> REPLACE
MECHANISM: Normalized coarse-spatial contrast residual head

HYPOTHESIS: Adding a zero-initialized class-specific head over horizontal, vertical, and diagonal 2×2 feature contrasts will exceed 9,334 correct predictions by preserving coarse positional evidence omitted by the successful global-statistics bypass.

INTENDED_EDIT: Add an affine-normalized 192-feature contrast head computed from adaptive 2×2 pooled trunk features, keeping the model below the 250,000-parameter ceiling.

EVIDENCE: The global-statistics residual head improved correctness from 9,314 to 9,334, showing that class-specific bypasses around the 30-feature bottleneck help; the earlier contrast proposal was not verified and therefore supplies no contrary accuracy evidence.

<<<<<<< SEARCH
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
=======
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.contrast_norm = nn.BatchNorm1d(64 * 3)
        self.contrast_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.contrast_head.weight)
        nn.init.zeros_(self.contrast_head.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        return self.classifier(feature_map) + residual_logits
=======
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        coarse = F.adaptive_avg_pool2d(feature_map, (2, 2))
        top_left = coarse[:, :, 0, 0]
        top_right = coarse[:, :, 0, 1]
        bottom_left = coarse[:, :, 1, 0]
        bottom_right = coarse[:, :, 1, 1]
        contrasts = torch.cat(
            (
                top_left + bottom_left - top_right - bottom_right,
                top_left + top_right - bottom_left - bottom_right,
                top_left + bottom_right - top_right - bottom_left,
            ),
            dim=1,
        )
        contrast_logits = self.contrast_head(
            self.contrast_norm(contrasts)
        )
        return (
            self.classifier(feature_map)
            + residual_logits
            + contrast_logits
        )
>>>>>>> REPLACE
MECHANISM: Normalized coarse-spatial residual bypass

HYPOTHESIS: Adding a zero-initialized class-specific head over normalized 2×2 pooled feature maps will exceed 9,334 correct predictions by preserving coarse spatial evidence discarded by the shared 30-feature bottleneck.

INTENDED_EDIT: Add a non-affine BatchNorm and direct ten-class residual head over four fixed spatial regions per channel, bringing the model to 249,988 learned parameters.

EVIDENCE: The normalized global-statistics residual improved correctness from 9,314 to 9,334, demonstrating that bypassing the shared bottleneck is beneficial; content-addressed pooling regressed, motivating a fixed-position rather than input-dependent complementary path.

<<<<<<< SEARCH
        self.statistics_norm = nn.BatchNorm1d(64 * 3)
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
=======
        self.statistics_norm = nn.BatchNorm1d(64 * 3)
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.coarse_norm = nn.BatchNorm1d(64 * 2 * 2, affine=False)
        self.coarse_head = nn.Linear(64 * 2 * 2, 10)
        nn.init.zeros_(self.coarse_head.weight)
        nn.init.zeros_(self.coarse_head.bias)
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
        coarse_features = F.adaptive_avg_pool2d(
            feature_map, output_size=2
        ).flatten(1)
        coarse_logits = self.coarse_head(
            self.coarse_norm(coarse_features)
        )
        return self.classifier(feature_map) + residual_logits + coarse_logits
>>>>>>> REPLACE
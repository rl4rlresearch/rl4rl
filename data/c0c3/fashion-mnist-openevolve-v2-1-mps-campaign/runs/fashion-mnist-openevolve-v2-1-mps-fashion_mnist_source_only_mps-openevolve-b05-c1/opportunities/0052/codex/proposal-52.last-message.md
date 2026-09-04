MECHANISM: Normalized coarse-spatial residual bypass

HYPOTHESIS: Adding a zero-initialized class-specific head over normalized 2×2 pooled feature maps will exceed 9,334 correct predictions by preserving coarse positional evidence discarded by the shared 30-feature bottleneck.

INTENDED_EDIT: Add a non-affine normalization and direct ten-class residual head over four fixed spatial regions per channel, increasing learned parameters from 247,418 to 249,988.

EVIDENCE: The normalized global-statistics bypass improved correctness from 9,314 to 9,334, while content-addressed pooling regressed. The prior coarse-spatial proposal was not verified, so it supplies no contrary accuracy evidence and merits a corrected direct test.

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
        self.spatial_norm = nn.BatchNorm1d(64 * 2 * 2, affine=False)
        self.spatial_head = nn.Linear(64 * 2 * 2, 10)
        nn.init.zeros_(self.spatial_head.weight)
        nn.init.zeros_(self.spatial_head.bias)
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
        spatial_evidence = F.adaptive_avg_pool2d(
            feature_map, output_size=(2, 2)
        ).flatten(1)
        spatial_logits = self.spatial_head(
            self.spatial_norm(spatial_evidence)
        )
        return self.classifier(feature_map) + residual_logits + spatial_logits
>>>>>>> REPLACE
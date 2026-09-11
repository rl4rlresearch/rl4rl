MECHANISM: Orthogonal coarse-spatial contrast residual head

HYPOTHESIS: A normalized class-specific head over horizontal, vertical, and diagonal 2×2 feature contrasts will exceed 9,334 correct predictions by adding coarse positional evidence without redundantly relearning the global mean already used by the successful statistics head.

INTENDED_EDIT: Add a zero-initialized residual classifier over three orthogonal contrasts from 2×2 average-pooled feature maps, bringing the model to 249,732 learned parameters.

EVIDENCE: The global-statistics bypass improved correctness from 9,314 to 9,334, showing that class-specific paths around the 30-feature bottleneck help. The two coarse-spatial proposals were not verified and provide no contrary accuracy result; this compact formulation removes their redundant pooled-mean component and avoids operating at the parameter ceiling.

<<<<<<< SEARCH
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
=======
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.spatial_contrast_norm = nn.BatchNorm1d(64 * 3)
        self.spatial_contrast_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.spatial_contrast_head.weight)
        nn.init.zeros_(self.spatial_contrast_head.bias)
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
        pooled = F.adaptive_avg_pool2d(feature_map, (2, 2))
        top_left = pooled[:, :, 0, 0]
        top_right = pooled[:, :, 0, 1]
        bottom_left = pooled[:, :, 1, 0]
        bottom_right = pooled[:, :, 1, 1]
        spatial_contrasts = torch.cat(
            (
                top_left + bottom_left - top_right - bottom_right,
                top_left + top_right - bottom_left - bottom_right,
                top_left + bottom_right - top_right - bottom_left,
            ),
            dim=1,
        )
        spatial_logits = self.spatial_contrast_head(
            self.spatial_contrast_norm(spatial_contrasts)
        )
        return self.classifier(feature_map) + residual_logits + spatial_logits
>>>>>>> REPLACE
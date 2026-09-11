MECHANISM: Flip-invariant class-specific spatial contrast head

HYPOTHESIS: Adding normalized vertical and magnitude-based quadrant contrasts will exceed 9,360 correct predictions by exposing coarse spatial layout directly to each class without exceeding 250,000 parameters.

INTENDED_EDIT: Add a zero-initialized 192-feature spatial residual head using top-bottom, absolute left-right, and absolute diagonal contrasts; total learned parameters become 249,860.

EVIDENCE: Agreement calibration from 0.04 through its fitted optimum of 0.31 repeatedly improved cross-entropy but never changed the 9,360 predictions, indicating that further scalar calibration is unlikely to improve the primary objective. The earlier full-quadrant spatial-head implementation could not be verified, so this lower-dimensional formulation tests the same capacity hypothesis with a simpler under-budget design.

<<<<<<< SEARCH
        self.statistics_norm = nn.BatchNorm1d(64 * 3)
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
=======
        self.statistics_norm = nn.BatchNorm1d(64 * 3)
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.spatial_norm = nn.BatchNorm1d(64 * 3)
        self.spatial_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.spatial_head.weight)
        nn.init.zeros_(self.spatial_head.bias)
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
>>>>>>> REPLACE

<<<<<<< SEARCH
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        return self.classifier(gated_feature_map) + residual_logits
=======
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        quadrants = F.adaptive_avg_pool2d(
            gated_feature_map, output_size=(2, 2)
        )
        vertical_contrast = (
            quadrants[:, :, 0, :].mean(dim=2)
            - quadrants[:, :, 1, :].mean(dim=2)
        )
        horizontal_contrast = (
            quadrants[:, :, :, 0].mean(dim=2)
            - quadrants[:, :, :, 1].mean(dim=2)
        ).abs()
        diagonal_contrast = (
            0.5
            * (
                quadrants[:, :, 0, 0]
                + quadrants[:, :, 1, 1]
                - quadrants[:, :, 0, 1]
                - quadrants[:, :, 1, 0]
            )
        ).abs()
        spatial_features = torch.cat(
            (
                vertical_contrast,
                horizontal_contrast,
                diagonal_contrast,
            ),
            dim=1,
        )
        spatial_logits = self.spatial_head(
            self.spatial_norm(spatial_features)
        )
        return (
            self.classifier(gated_feature_map)
            + residual_logits
            + spatial_logits
        )
>>>>>>> REPLACE
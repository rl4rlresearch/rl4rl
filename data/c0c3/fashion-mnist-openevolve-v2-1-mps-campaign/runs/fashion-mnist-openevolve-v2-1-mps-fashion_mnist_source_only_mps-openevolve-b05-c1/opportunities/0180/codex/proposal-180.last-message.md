MECHANISM: Class-conditional coarse spatial residual head

HYPOTHESIS: Bypassing the shared 30-unit positional bottleneck with normalized, class-specific 2×2 spatial evidence will exceed 9,360 correct predictions by distinguishing classes through coarse part layout.

INTENDED_EDIT: Add a zero-initialized residual head over normalized quadrant-pooled feature maps; make the existing statistics BatchNorm affine-free to remain below the parameter ceiling.

EVIDENCE: Agreement calibration repeatedly lowered cross-entropy without changing 9,360 decisions, while loss and dropout changes reduced accuracy; this challenges the assumption that confidence or regularization is limiting and instead tests new learned class-prediction capacity. The poorer diagonal-translation result also suggests preserving spatial layout is useful.

<<<<<<< SEARCH
        self.statistics_norm = nn.BatchNorm1d(64 * 3)
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
=======
        self.statistics_norm = nn.BatchNorm1d(64 * 3, affine=False)
        self.statistics_head = nn.Linear(64 * 3, 10)
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.spatial_norm = nn.BatchNorm1d(64 * 2 * 2, affine=False)
        self.spatial_head = nn.Linear(64 * 2 * 2, 10, bias=False)
        nn.init.zeros_(self.spatial_head.weight)
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
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
        coarse_spatial = F.adaptive_avg_pool2d(
            gated_feature_map, (2, 2)
        ).flatten(1)
        spatial_logits = self.spatial_head(
            self.spatial_norm(coarse_spatial)
        )
        return (
            self.classifier(gated_feature_map)
            + residual_logits
            + spatial_logits
        )
>>>>>>> REPLACE
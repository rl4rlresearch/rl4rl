MECHANISM: Identity-initialized spatial recalibration

HYPOTHESIS: A learned content-conditioned spatial gate will exceed 9,360 correct predictions by suppressing background activations and emphasizing discriminative regions before both prediction heads.

INTENDED_EDIT: Add a 99-parameter, 7×7 spatial-attention gate derived from channel mean and maximum maps, initialized to preserve the current model exactly.

EVIDENCE: Center-weight refinements plateaued at 9,360 correct with only tiny cross-entropy gains, suggesting inference aggregation is saturated; this challenges the shared assumption that global channel gating is sufficient and every spatial location should enter the learned representation unconditionally.

<<<<<<< SEARCH
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
=======
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
        self.spatial_recalibration = nn.Conv2d(
            2, 1, kernel_size=7, padding=3
        )
        nn.init.zeros_(self.spatial_recalibration.weight)
        nn.init.zeros_(self.spatial_recalibration.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(images)
        spatial_mean = feature_map.mean(dim=(2, 3))
        spatial_std = feature_map.var(
            dim=(2, 3), unbiased=False
        ).add(1e-4).sqrt()
        spatial_max = feature_map.amax(dim=(2, 3))
        channel_gate = 2.0 * torch.sigmoid(
            spatial_mean * self.recalibration_scale
            + self.recalibration_bias
        )
        gated_feature_map = (
            feature_map * channel_gate[:, :, None, None]
        )
        statistics = torch.cat(
            (spatial_mean, spatial_std, spatial_max), dim=1
        )
        statistics = statistics * channel_gate.repeat(1, 3)
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        return self.classifier(gated_feature_map) + residual_logits
=======
    def _forward_once(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.features(images)
        channel_summary = feature_map.mean(dim=(2, 3))
        channel_gate = 2.0 * torch.sigmoid(
            channel_summary * self.recalibration_scale
            + self.recalibration_bias
        )
        spatial_descriptor = torch.cat(
            (
                feature_map.mean(dim=1, keepdim=True),
                feature_map.amax(dim=1, keepdim=True),
            ),
            dim=1,
        )
        spatial_gate = 2.0 * torch.sigmoid(
            self.spatial_recalibration(spatial_descriptor)
        )
        spatially_gated_feature_map = feature_map * spatial_gate
        gated_feature_map = (
            spatially_gated_feature_map
            * channel_gate[:, :, None, None]
        )
        spatial_mean = spatially_gated_feature_map.mean(dim=(2, 3))
        spatial_std = spatially_gated_feature_map.var(
            dim=(2, 3), unbiased=False
        ).add(1e-4).sqrt()
        spatial_max = spatially_gated_feature_map.amax(dim=(2, 3))
        statistics = torch.cat(
            (spatial_mean, spatial_std, spatial_max), dim=1
        )
        statistics = statistics * channel_gate.repeat(1, 3)
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        return self.classifier(gated_feature_map) + residual_logits
>>>>>>> REPLACE
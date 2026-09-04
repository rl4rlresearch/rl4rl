MECHANISM: Diagonal global-context feature recalibration

HYPOTHESIS: A per-channel self-conditioned gate will exceed 9,344 correct predictions by retaining the beneficial full-feature-map recalibration signal while avoiding the runtime cost of a squeeze-and-excitation MLP.

INTENDED_EDIT: Add an identity-initialized, 128-parameter gate derived from each channel’s global mean and apply it analytically to both the feature map and its mean/std/max statistics.

EVIDENCE: Full-map squeeze-and-excitation reached 9,345 correct—the only representation change above 9,344—but exceeded the time limit, while bottleneck-only gating tied at 9,344. This suggests modulation must occur before the spatial classifier; a diagonal gate tests that signal with substantially less computation.

<<<<<<< SEARCH
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
=======
        nn.init.zeros_(self.statistics_head.weight)
        nn.init.zeros_(self.statistics_head.bias)
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
>>>>>>> REPLACE

<<<<<<< SEARCH
        spatial_max = feature_map.amax(dim=(2, 3))
        statistics = torch.cat(
            (spatial_mean, spatial_std, spatial_max), dim=1
        )
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
        return self.classifier(feature_map) + residual_logits
=======
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
>>>>>>> REPLACE
MECHANISM: Compact global-context channel recalibration

HYPOTHESIS: An eight-dimensional squeeze-and-excitation gate will exceed 9,344 correct predictions while remaining within the time limit by retaining the promising global-context recalibration signal with roughly half the added parameters and a fused variance/mean reduction.

INTENDED_EDIT: Add an identity-initialized low-rank channel gate, apply it to trunk features and their statistics, and compute variance and mean jointly.

EVIDENCE: The 16-dimensional channel-recalibration design reported 9,345 correct—the only tested representation change above the current 9,344—but took 79.6 seconds and timed out; a narrower gate directly tests that signal with less computation.

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
        self.channel_gate = nn.Sequential(
            nn.Linear(64, 8),
            nn.GELU(),
            nn.Linear(8, 64),
        )
        nn.init.zeros_(self.channel_gate[-1].weight)
        nn.init.zeros_(self.channel_gate[-1].bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        feature_map = self.features(images)
        spatial_mean = feature_map.mean(dim=(2, 3))
        spatial_std = feature_map.var(
            dim=(2, 3), unbiased=False
        ).add(1e-4).sqrt()
        spatial_max = feature_map.amax(dim=(2, 3))
        statistics = torch.cat(
            (spatial_mean, spatial_std, spatial_max), dim=1
        )
=======
        feature_map = self.features(images)
        spatial_variance, spatial_mean = torch.var_mean(
            feature_map, dim=(2, 3), unbiased=False
        )
        spatial_max = feature_map.amax(dim=(2, 3))
        channel_scale = 1.0 + torch.tanh(
            self.channel_gate(spatial_mean)
        )
        feature_map = feature_map * channel_scale[:, :, None, None]
        spatial_mean = spatial_mean * channel_scale
        spatial_std = spatial_variance.add(1e-4).sqrt() * channel_scale
        spatial_max = spatial_max * channel_scale
        statistics = torch.cat(
            (spatial_mean, spatial_std, spatial_max), dim=1
        )
>>>>>>> REPLACE
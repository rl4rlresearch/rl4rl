MECHANISM: Vertical-contrast statistics bypass

HYPOTHESIS: Adding a per-channel bottom-versus-top activation contrast will exceed 9,345 correct predictions by supplying coarse spatial layout directly to the classifier without relying on the rank-30 flattened bottleneck.

INTENDED_EDIT: Extend the normalized statistics residual from mean/std/max to mean/std/max/vertical-contrast, adding 768 parameters for a total of 248,314 while preserving the initial logits.

EVIDENCE: The statistics bypass improved prior results, while attempts to widen positional processing timed out; a single low-frequency spatial statistic targets the same lost spatial information with substantially less computation and parameter cost.

<<<<<<< SEARCH
        self.statistics_norm = nn.BatchNorm1d(64 * 3)
        self.statistics_head = nn.Linear(64 * 3, 10)
=======
        self.statistics_norm = nn.BatchNorm1d(64 * 4)
        self.statistics_head = nn.Linear(64 * 4, 10)
>>>>>>> REPLACE

<<<<<<< SEARCH
        statistics = torch.cat(
            (spatial_mean, spatial_std, spatial_max), dim=1
        )
        statistics = statistics * channel_gate.repeat(1, 3)
=======
        vertical_contrast = (
            feature_map[:, :, 4:, :].mean(dim=(2, 3))
            - feature_map[:, :, :3, :].mean(dim=(2, 3))
        )
        statistics = torch.cat(
            (
                spatial_mean,
                spatial_std,
                spatial_max,
                vertical_contrast,
            ),
            dim=1,
        )
        statistics = statistics * channel_gate.repeat(1, 4)
>>>>>>> REPLACE
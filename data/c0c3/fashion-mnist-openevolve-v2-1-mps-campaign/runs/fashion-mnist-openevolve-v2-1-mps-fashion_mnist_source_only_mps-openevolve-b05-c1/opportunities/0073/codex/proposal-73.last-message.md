MECHANISM: Spatial-path-only diagonal recalibration

HYPOTHESIS: Restricting the channel gate to the spatial classifier will exceed 9,345 correct predictions by preserving the successful feature-map modulation while preventing sample-dependent scaling from destabilizing the BatchNorm-normalized statistics branch.

INTENDED_EDIT: Stop applying the learned channel gate to the mean/std/max statistics; retain it unchanged on the full feature map.

EVIDENCE: Raw-mean diagonal gating improved the best result from 9,344 to 9,345, while bottleneck-only gating tied at 9,344 and normalized gate conditioning regressed to 9,318. This directly tests whether the gain comes from pre-classifier spatial modulation while keeping the statistics residual on its established normalization distribution.

<<<<<<< SEARCH
        statistics = torch.cat(
            (spatial_mean, spatial_std, spatial_max), dim=1
        )
        statistics = statistics * channel_gate.repeat(1, 3)
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
=======
        statistics = torch.cat(
            (spatial_mean, spatial_std, spatial_max), dim=1
        )
        residual_logits = self.statistics_head(
            self.statistics_norm(statistics)
        )
>>>>>>> REPLACE
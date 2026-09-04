MECHANISM: Channel-wise peak-prominence statistic

HYPOTHESIS: Adding each channel’s standardized peak prominence will exceed 9,342 correct predictions by distinguishing localized activations from diffuse responses using statistics already computed by the successful residual head.

INTENDED_EDIT: Extend the normalized statistics head from three to four features per channel with `(maximum - mean) / standard_deviation`, increasing learned parameters from 247,418 to 248,186 without another spatial reduction.

EVIDENCE: The mean/std/max statistics bypass improved correctness from 9,314 to 9,334 and EMA raised it to 9,342, while additional spatial computations exceeded the time limit; this adds complementary nonlinear shape information through the proven bypass at negligible computational cost.

<<<<<<< SEARCH
        self.statistics_norm = nn.BatchNorm1d(64 * 3)
        self.statistics_head = nn.Linear(64 * 3, 10)
=======
        self.statistics_norm = nn.BatchNorm1d(64 * 4)
        self.statistics_head = nn.Linear(64 * 4, 10)
>>>>>>> REPLACE

<<<<<<< SEARCH
        spatial_max = feature_map.amax(dim=(2, 3))
        statistics = torch.cat(
            (spatial_mean, spatial_std, spatial_max), dim=1
        )
=======
        spatial_max = feature_map.amax(dim=(2, 3))
        peak_prominence = (
            spatial_max - spatial_mean
        ) / spatial_std
        statistics = torch.cat(
            (
                spatial_mean,
                spatial_std,
                spatial_max,
                peak_prominence,
            ),
            dim=1,
        )
>>>>>>> REPLACE
MECHANISM: Channel-wise spatial total-variation statistic

HYPOTHESIS: Extending the successful global-statistics bypass with per-channel local variation will exceed 9,334 correct predictions by capturing feature-map edge and texture structure that mean, standard deviation, and maximum do not encode.

INTENDED_EDIT: Add mean absolute horizontal/vertical feature differences as a fourth normalized statistic while retaining the same zero-initialized residual head design; learned parameters increase from 247,418 to 248,186.

EVIDENCE: The normalized global-statistics bypass improved correctness from 9,314 to 9,334, while more expensive positional bypasses could not be verified; this tests a compact complementary signal within the proven path.

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
        horizontal_variation = (
            feature_map[:, :, :, 1:] - feature_map[:, :, :, :-1]
        ).abs().mean(dim=(2, 3))
        vertical_variation = (
            feature_map[:, :, 1:, :] - feature_map[:, :, :-1, :]
        ).abs().mean(dim=(2, 3))
        spatial_variation = 0.5 * (
            horizontal_variation + vertical_variation
        )
        statistics = torch.cat(
            (
                spatial_mean,
                spatial_std,
                spatial_max,
                spatial_variation,
            ),
            dim=1,
        )
>>>>>>> REPLACE
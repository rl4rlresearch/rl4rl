MECHANISM: Learned spatial-moment residual classification

HYPOTHESIS: Adding horizontal, vertical, and radial feature moments to the auxiliary head will exceed 9,360 correct predictions by directly exposing coarse silhouette layout while remaining below the parameter ceiling at 249,850 parameters.

INTENDED_EDIT: Expand the orderless global-statistics branch from three to six statistics per channel using fixed spatial-moment bases and a learned class-logit head.

EVIDENCE: Confidence-calibration changes repeatedly lowered cross-entropy without changing any of the 9,360 decisions, indicating that representation—not confidence—is load-bearing. The old branch assumes spatially orderless mean/std/max statistics suffice; this patch tests whether compact spatial layout can resolve additional decisions without the computational cost of learned pooling.

<<<<<<< SEARCH
        self.statistics_norm = nn.BatchNorm1d(64 * 3)
        self.statistics_head = nn.Linear(64 * 3, 10)
=======
        axis = torch.linspace(-1.0, 1.0, 7)
        grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
        radial = grid_x.square() + grid_y.square()
        radial = radial - radial.mean()
        self.register_buffer(
            "spatial_moment_weights",
            torch.stack((grid_x, grid_y, radial)),
        )
        self.statistics_norm = nn.BatchNorm1d(64 * 6)
        self.statistics_head = nn.Linear(64 * 6, 10)
>>>>>>> REPLACE

<<<<<<< SEARCH
        statistics = torch.cat(
            (spatial_mean, spatial_std, spatial_max), dim=1
        )
        statistics = statistics * channel_gate.repeat(1, 3)
=======
        spatial_moments = (
            feature_map[:, :, None]
            * self.spatial_moment_weights[None, None]
        ).mean(dim=(3, 4))
        moment_x, moment_y, moment_radial = spatial_moments.unbind(dim=2)
        statistics = torch.cat(
            (
                spatial_mean,
                spatial_std,
                spatial_max,
                moment_x,
                moment_y,
                moment_radial,
            ),
            dim=1,
        )
        statistics = statistics * channel_gate.repeat(1, 6)
>>>>>>> REPLACE
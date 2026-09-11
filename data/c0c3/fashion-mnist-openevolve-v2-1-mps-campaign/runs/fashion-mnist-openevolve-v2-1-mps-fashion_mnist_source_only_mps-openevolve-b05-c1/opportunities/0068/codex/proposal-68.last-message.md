MECHANISM: Rank-one cross-channel feature recalibration

HYPOTHESIS: A learned rank-one summary of all channel means will exceed 9,345 correct predictions by adding cross-channel context to the successful diagonal gate without the runtime cost of squeeze-and-excitation.

INTENDED_EDIT: Extend the identity-initialized gate with a learned 64-to-1 context projection and per-channel response, while fusing the existing mean and variance reduction; parameters increase from 247,546 to 247,674.

EVIDENCE: Full squeeze-and-excitation and diagonal gating both reached 9,345 correct, but the cross-channel design exceeded the time limit; a rank-one interaction tests whether its missing cross-channel signal can improve the efficient diagonal design.

<<<<<<< SEARCH
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
=======
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
        self.context_projection = nn.Parameter(torch.full((64,), 1.0 / 64.0))
        self.context_scale = nn.Parameter(torch.zeros(64))
>>>>>>> REPLACE

<<<<<<< SEARCH
        spatial_mean = feature_map.mean(dim=(2, 3))
        spatial_std = feature_map.var(
            dim=(2, 3), unbiased=False
        ).add(1e-4).sqrt()
        spatial_max = feature_map.amax(dim=(2, 3))
        channel_gate = 2.0 * torch.sigmoid(
            spatial_mean * self.recalibration_scale
            + self.recalibration_bias
        )
=======
        spatial_variance, spatial_mean = torch.var_mean(
            feature_map, dim=(2, 3), unbiased=False
        )
        spatial_std = spatial_variance.add(1e-4).sqrt()
        spatial_max = feature_map.amax(dim=(2, 3))
        global_context = spatial_mean.matmul(
            self.context_projection
        ).unsqueeze(1)
        channel_gate = 2.0 * torch.sigmoid(
            spatial_mean * self.recalibration_scale
            + global_context * self.context_scale
            + self.recalibration_bias
        )
>>>>>>> REPLACE
MECHANISM: Multi-statistic diagonal channel recalibration

HYPOTHESIS: Conditioning the identity-initialized channel gate on raw mean, standard deviation, and maximum will exceed 9,345 correct predictions by learning complementary per-channel distribution cues without the harmful normalization or runtime cost seen in prior recalibration variants.

INTENDED_EDIT: Add zero-initialized per-channel scale vectors for the already-computed spatial standard deviation and maximum, then include them in the existing gate.

EVIDENCE: Raw-mean diagonal gating produced the current best 9,345 correct, whereas standardized and running-mean-centered conditioning regressed; extending the successful raw-scale gate with already-available descriptors isolates whether its conditioning signal is under-specified.

<<<<<<< SEARCH
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
=======
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_std_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_max_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
>>>>>>> REPLACE

<<<<<<< SEARCH
        channel_gate = 2.0 * torch.sigmoid(
            spatial_mean * self.recalibration_scale
            + self.recalibration_bias
        )
=======
        channel_gate = 2.0 * torch.sigmoid(
            spatial_mean * self.recalibration_scale
            + spatial_std * self.recalibration_std_scale
            + spatial_max * self.recalibration_max_scale
            + self.recalibration_bias
        )
>>>>>>> REPLACE
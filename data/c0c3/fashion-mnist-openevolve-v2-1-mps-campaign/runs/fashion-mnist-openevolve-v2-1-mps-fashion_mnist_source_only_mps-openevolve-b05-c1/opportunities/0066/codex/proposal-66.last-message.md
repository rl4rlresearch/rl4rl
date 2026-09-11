MECHANISM: Standardized diagonal channel recalibration

HYPOTHESIS: Standardizing each channel’s global mean before self-gating will exceed 9,345 correct predictions by separating image-dependent variation from the gate’s static bias without adding learned parameters or material runtime.

INTENDED_EDIT: Add a parameter-free BatchNorm context normalizer and use its standardized channel means to drive the existing identity-initialized gate.

EVIDENCE: Raw-mean diagonal gating improved the best result from 9,344 to 9,345, while bottleneck gating only tied and heavier full-map gates timed out; improving the successful pre-classifier gate’s conditioning is therefore the most direct low-cost refinement.

<<<<<<< SEARCH
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
=======
        self.recalibration_norm = nn.BatchNorm1d(64, affine=False)
        self.recalibration_scale = nn.Parameter(torch.zeros(64))
        self.recalibration_bias = nn.Parameter(torch.zeros(64))
>>>>>>> REPLACE

<<<<<<< SEARCH
        channel_gate = 2.0 * torch.sigmoid(
            spatial_mean * self.recalibration_scale
            + self.recalibration_bias
        )
=======
        recalibration_context = self.recalibration_norm(spatial_mean)
        channel_gate = 2.0 * torch.sigmoid(
            recalibration_context * self.recalibration_scale
            + self.recalibration_bias
        )
>>>>>>> REPLACE
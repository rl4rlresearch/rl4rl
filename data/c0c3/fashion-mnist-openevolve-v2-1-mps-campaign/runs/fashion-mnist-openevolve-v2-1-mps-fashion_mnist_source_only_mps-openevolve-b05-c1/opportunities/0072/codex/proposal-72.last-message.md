MECHANISM: Running-mean-centered diagonal channel recalibration

HYPOTHESIS: Centering each channel’s raw mean with the existing statistics BatchNorm running mean will exceed 9,345 correct predictions by separating image-dependent gate variation from static channel bias without the harmful variance normalization observed previously.

INTENDED_EDIT: Drive the existing identity-initialized gate with running-mean-centered channel means, adding no parameters, reductions, or material runtime.

EVIDENCE: Raw-mean gating improved the best result to 9,345, while full BatchNorm standardization fell to 9,318; retaining raw scale while removing only the learned baseline is the most direct test of whether variance normalization caused that regression.

<<<<<<< SEARCH
        channel_gate = 2.0 * torch.sigmoid(
            spatial_mean * self.recalibration_scale
            + self.recalibration_bias
        )
=======
        centered_mean = (
            spatial_mean
            - self.statistics_norm.running_mean[:64]
        )
        channel_gate = 2.0 * torch.sigmoid(
            centered_mean * self.recalibration_scale
            + self.recalibration_bias
        )
>>>>>>> REPLACE
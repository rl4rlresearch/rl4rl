MECHANISM: Coordinate-swapped fourth post-attention LayerNorm bias reduction

HYPOTHESIS: Retaining the fifth active `ln2` bias coordinate instead of the fourth will reduce the model to 1610 parameters while maintaining at least 99% accuracy, indicating that the prior four-contrast failure was coordinate-specific.

INTENDED_EDIT: Store four learned `ln2` bias contrasts and place them in the first three and fifth active coordinates before centering.

EVIDENCE: A coordinate swap rescued the analogous two-contrast `ln1` reduction from 96.77% to 99.97%; this directly motivates testing an alternative contrast basis for the previously failed fourth `ln2` reduction.

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.zeros(d_model - 3))
=======
        self.bias = nn.Parameter(torch.zeros(d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        active_bias = torch.cat((self.bias, self.bias.new_zeros(2)))
=======
        active_bias = torch.cat(
            (
                self.bias[:3],
                self.bias.new_zeros(1),
                self.bias[3:],
                self.bias.new_zeros(2),
            )
        )
>>>>>>> REPLACE
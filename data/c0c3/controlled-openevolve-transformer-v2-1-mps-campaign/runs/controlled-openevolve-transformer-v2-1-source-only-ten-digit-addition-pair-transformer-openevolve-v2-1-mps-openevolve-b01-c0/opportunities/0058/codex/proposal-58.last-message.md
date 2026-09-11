MECHANISM: Coordinate-selected three-contrast post-attention LayerNorm bias

HYPOTHESIS: Retaining the first, third, and fifth active `ln2` bias coordinates will reduce the model to 1609 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Remove the second-coordinate `ln2` bias degree of freedom while preserving learned contrasts in the first, third, and fifth active coordinates before centering.

EVIDENCE: Swapping the fourth `ln2` coordinate for the fifth raised the 1610-parameter design to 99.77%, and the analogous `ln1` evidence showed that retaining the third rather than second coordinate rescued a reduced-contrast design from 96.77% to 99.97%; this motivates preserving the third and fifth coordinates while testing removal of the second.

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.zeros(d_model - 4))
=======
        self.bias = nn.Parameter(torch.zeros(d_model - 5))
>>>>>>> REPLACE

<<<<<<< SEARCH
        active_bias = torch.cat(
            (
                self.bias[:3],
                self.bias.new_zeros(1),
                self.bias[3:],
                self.bias.new_zeros(2),
            )
        )
=======
        active_bias = torch.cat(
            (
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:2],
                self.bias.new_zeros(1),
                self.bias[2:],
                self.bias.new_zeros(2),
            )
        )
>>>>>>> REPLACE
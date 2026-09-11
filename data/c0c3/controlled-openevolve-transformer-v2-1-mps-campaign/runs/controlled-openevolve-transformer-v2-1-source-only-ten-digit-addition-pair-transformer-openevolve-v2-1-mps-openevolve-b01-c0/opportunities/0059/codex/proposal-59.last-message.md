MECHANISM: Third-and-fifth-coordinate post-attention LayerNorm bias

HYPOTHESIS: Retaining only the third and fifth active `ln2` bias coordinates will reduce the model to 1608 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Remove the first-coordinate `ln2` bias degree of freedom and reconstruct the seven active coordinates from two learned contrasts in positions three and five.

EVIDENCE: The fifth coordinate rescued the four-contrast `ln2` design to 99.77%, while the third coordinate was the strongest reduced-bias coordinate for `ln1`; the current first/third/fifth design reaches 99.86%, making removal of the less-supported first coordinate the most informative next reduction.

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.zeros(d_model - 5))
=======
        self.bias = nn.Parameter(torch.zeros(d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        active_bias = torch.cat(
            (
                self.bias.new_zeros(2),
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
                self.bias.new_zeros(2),
            )
        )
>>>>>>> REPLACE
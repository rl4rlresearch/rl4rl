MECHANISM: Third-coordinate-only post-attention LayerNorm bias

HYPOTHESIS: Retaining the third-coordinate `ln2` contrast while removing the fifth will reduce the model to 1607 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Store one learned `ln2` bias contrast and place it only in the third active coordinate before centering.

EVIDENCE: The complementary fifth-only design scored 94.97%, while the third coordinate was the strongest reduced-bias coordinate for `ln1`; testing third-only isolates whether it supplies the essential contribution in the 99.90% third/fifth design.

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.zeros(d_model - 6))
=======
        self.bias = nn.Parameter(torch.zeros(d_model - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
        active_bias = torch.cat(
            (
                self.bias.new_zeros(2),
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
                self.bias.new_zeros(2),
            )
        )
=======
        active_bias = torch.cat(
            (
                self.bias.new_zeros(2),
                self.bias,
                self.bias.new_zeros(4),
            )
        )
>>>>>>> REPLACE
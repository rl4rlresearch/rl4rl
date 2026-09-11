MECHANISM: Fifth-coordinate-only post-attention LayerNorm bias

HYPOTHESIS: Removing the third-coordinate `ln2` contrast while retaining the fifth will reduce the model to 1607 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Store one learned `ln2` bias contrast and place it in the fifth active coordinate before centering.

EVIDENCE: Adding the fifth coordinate rescued the reduced `ln2` design to 99.77%, and the current third/fifth design reaches 99.90%; this makes the less directly supported third coordinate the most informative next removal.

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
                self.bias.new_zeros(4),
                self.bias,
                self.bias.new_zeros(2),
            )
        )
>>>>>>> REPLACE
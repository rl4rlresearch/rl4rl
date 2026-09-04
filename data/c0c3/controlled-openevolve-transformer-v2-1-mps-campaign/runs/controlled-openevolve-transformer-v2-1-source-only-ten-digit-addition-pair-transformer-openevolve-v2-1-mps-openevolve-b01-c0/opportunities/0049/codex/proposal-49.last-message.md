MECHANISM: Third-coordinate-only pre-attention LayerNorm bias

HYPOTHESIS: Retaining only the third active `ln1` bias coordinate will reduce the model to 1613 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Remove the first learned `ln1` bias contrast and preserve the third-coordinate contrast before centering.

EVIDENCE: Two contrasts in the first and third coordinates achieved 99.97% at 1614 parameters, while first and second achieved 96.77%; this identifies the third coordinate as the strongest candidate to retain in a one-contrast design.

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.zeros(d_model - 6))
=======
        self.bias = nn.Parameter(torch.zeros(d_model - 7))
>>>>>>> REPLACE

<<<<<<< SEARCH
        active_bias = torch.cat(
            (
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
                self.bias.new_zeros(4),
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
MECHANISM: Second LayerNorm-bias/downstream-affine gauge

HYPOTHESIS: Fixing second-LayerNorm bias coordinate 1 will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because this bias contribution is absorbable by the learned `fc1` bias and feature 1 has previously supported 99.98% value-bias and 99.81% LayerNorm-scale reductions.

INTENDED_EDIT: Remove LayerNorm-2 bias coordinate 1 in addition to the existing fixed coordinate 4, reconstructing both as zeros during the forward pass.

EVIDENCE: The verified 1616-parameter design already succeeds with LayerNorm-2 bias coordinate 4 fixed. Among alternative coordinates, feature 1 has the strongest stated affine-coordinate evidence, while recent feature-1 failure concerned the more optimization-sensitive token-position embedding quotient.

<<<<<<< SEARCH
        self.fixed_index = 4
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))
=======
        self.fixed_index = 4
        self.fixed_bias_indices = (1, 4)
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_bias = torch.cat(
            (
                self.bias[: self.fixed_index],
                self.bias.new_zeros(1),
                self.bias[self.fixed_index :],
            )
        )
=======
        full_bias = torch.cat(
            (
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:3],
                self.bias.new_zeros(1),
                self.bias[3:],
            )
        )
>>>>>>> REPLACE
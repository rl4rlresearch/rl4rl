MECHANISM: Incremental value-bias coordinate anchoring

HYPOTHESIS: Retaining six of eight learned value-bias coordinates will reduce the model to 1589 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Fix a second value-bias coordinate to zero, preserving all attention weights, mean-zero residual parameterizations, MLP capacity, training, and decoding behavior.

EVIDENCE: The current seven-coordinate design achieved 99.94% accuracy with 1590 parameters, while removing all value-bias coordinates failed; this motivates the smallest incremental ablation.

<<<<<<< SEARCH
        self.v_bias = nn.Parameter(torch.zeros(d_model - 1))
=======
        self.v_bias = nn.Parameter(torch.zeros(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_bias = F.pad(self.v_bias, (0, 1))
=======
        value_bias = F.pad(self.v_bias, (0, 2))
>>>>>>> REPLACE
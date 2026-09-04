MECHANISM: Dense rotated pre-attention LayerNorm bias quotient

HYPOTHESIS: Replacing the failed high-index `ln1.bias` truncation with a balanced five-dimensional subspace will produce a 1,563-parameter model with at least 99% accuracy by distributing the redundant constraint across all six previously qualified coordinates.

INTENDED_EDIT: Reduce `ln1.bias` from six to five learned coordinates, using an orthonormal rotation that removes the equal combination of the six qualified basis directions; existing full-coordinate AdamW handling remains compatible.

EVIDENCE: The six-coordinate `ln1.bias` model achieved 99.86% at 1,564 parameters, while simply dropping its highest-index coordinate collapsed to 60.95%; this tests whether that failure was caused by the particular anisotropic coordinate removal rather than the exact pre-attention bias redundancy.

<<<<<<< SEARCH
        # Constant pre-attention shifts are absorbable by the query and
        # output-projection biases; their key shifts cancel from the softmax.
        # Retain six coordinates while quotienting one additional direction.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 2))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 2)
        for col in range(cfg.d_model - 2):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln1_bias_basis", ln1_bias_basis, persistent=False)
=======
        # Constant pre-attention shifts are absorbable by the query and
        # output-projection biases; their key shifts cancel from the softmax.
        # Remove a dense combination of the six previously qualified
        # coordinates instead of truncating its optimization-sensitive edge.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 3))
        ln1_full_basis = torch.zeros(cfg.d_model, cfg.d_model - 2)
        for col in range(cfg.d_model - 2):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_full_basis[: col + 1, col] = 1.0 / scale
            ln1_full_basis[col + 1, col] = -(col + 1) / scale

        ln1_reduction = torch.zeros(cfg.d_model - 2, cfg.d_model - 3)
        for col in range(cfg.d_model - 3):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_reduction[: col + 1, col] = 1.0 / scale
            ln1_reduction[col + 1, col] = -(col + 1) / scale
        ln1_bias_basis = ln1_full_basis @ ln1_reduction
        self.register_buffer("ln1_bias_basis", ln1_bias_basis, persistent=False)
>>>>>>> REPLACE
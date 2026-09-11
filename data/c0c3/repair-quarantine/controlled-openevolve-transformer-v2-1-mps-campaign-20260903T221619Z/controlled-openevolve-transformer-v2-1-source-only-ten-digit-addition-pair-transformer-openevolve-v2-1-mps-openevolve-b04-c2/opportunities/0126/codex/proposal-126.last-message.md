MECHANISM: Nested balanced post-attention LayerNorm bias quotient

HYPOTHESIS: Reproducing the qualified 1,562-parameter design and removing one balanced direction from its three-dimensional `ln2.bias` subspace will yield 1,561 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Use the qualified five-coordinate balanced `ln1.bias` basis and replace the qualified three-coordinate balanced `ln2.bias` basis with a nested balanced two-coordinate basis; existing full-coordinate AdamW handling remains compatible.

EVIDENCE: Balanced reductions recovered 99.87% at 1,563 parameters for `ln1.bias` and 99.75% at 1,562 for `ln2.bias` after direct coordinate truncations failed, motivating one further distributed reduction of the same qualified `ln2` subspace.

<<<<<<< SEARCH
        # A uniform pre-attention shift is absorbable by the query bias and
        # output-projection bias; key shifts cancel from attention softmax.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 1))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln1_bias_basis", ln1_bias_basis, persistent=False)
=======
        # Remove a balanced combination of the six-coordinate subspace that
        # qualified, avoiding the optimization-sensitive direct truncation.
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

<<<<<<< SEARCH
        # Every constant post-normalization shift is absorbable by the
        # unrestricted fc1 bias. Remove one additional direction while
        # retaining six coordinates for nearly unchanged optimization.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 2))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 2)
        for col in range(cfg.d_model - 2):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_bias_basis[: col + 1, col] = 1.0 / scale
            ln2_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln2_bias_basis", ln2_bias_basis, persistent=False)
=======
        # Constant post-normalization shifts are absorbable by the
        # unrestricted fc1 bias. Starting from the qualified balanced
        # three-coordinate basis, remove one more distributed direction.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 6))
        ln2_full_basis = torch.zeros(cfg.d_model, cfg.d_model - 4)
        for col in range(cfg.d_model - 4):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_full_basis[: col + 1, col] = 1.0 / scale
            ln2_full_basis[col + 1, col] = -(col + 1) / scale

        ln2_reduction = torch.zeros(cfg.d_model - 4, cfg.d_model - 5)
        for col in range(cfg.d_model - 5):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_reduction[: col + 1, col] = 1.0 / scale
            ln2_reduction[col + 1, col] = -(col + 1) / scale

        ln2_second_reduction = torch.zeros(
            cfg.d_model - 5, cfg.d_model - 6
        )
        for col in range(cfg.d_model - 6):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_second_reduction[: col + 1, col] = 1.0 / scale
            ln2_second_reduction[col + 1, col] = -(col + 1) / scale
        ln2_bias_basis = (
            ln2_full_basis @ ln2_reduction @ ln2_second_reduction
        )
        self.register_buffer("ln2_bias_basis", ln2_bias_basis, persistent=False)
>>>>>>> REPLACE
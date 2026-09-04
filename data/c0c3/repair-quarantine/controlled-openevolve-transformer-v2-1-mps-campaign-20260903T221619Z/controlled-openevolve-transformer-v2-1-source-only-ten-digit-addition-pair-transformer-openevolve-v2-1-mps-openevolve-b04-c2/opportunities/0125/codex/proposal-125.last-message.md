MECHANISM: Dense rotated post-attention LayerNorm bias quotient

HYPOTHESIS: Replacing the failed high-index `ln2.bias` truncation with a balanced three-dimensional subspace, while using the qualified balanced five-dimensional `ln1.bias` subspace, will produce a 1,562-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified 1,563-parameter design, then remove one dense combination of the four qualified `ln2.bias` coordinates while retaining full-coordinate AdamW updates.

EVIDENCE: A balanced `ln1.bias` reduction recovered 99.87% accuracy at 1,563 parameters after direct coordinate truncation collapsed to 60.95%; this motivates testing whether the failed 98.62% high-index `ln2.bias` reduction was likewise caused by an anisotropic coordinate choice.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 1))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln1_bias_basis", ln1_bias_basis, persistent=False)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model)

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
        # Constant post-normalization shifts are absorbable by the
        # unrestricted fc1 bias. Retain five coordinates to test the next
        # progressive quotient beyond the qualified six-coordinate design.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 3))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 3)
        for col in range(cfg.d_model - 3):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_bias_basis[: col + 1, col] = 1.0 / scale
            ln2_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln2_bias_basis", ln2_bias_basis, persistent=False)
=======
        # Constant post-normalization shifts are absorbable by the
        # unrestricted fc1 bias. Remove a balanced combination of the four
        # qualified coordinates instead of truncating their high-index edge.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 5))
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
        ln2_bias_basis = ln2_full_basis @ ln2_reduction
        self.register_buffer("ln2_bias_basis", ln2_bias_basis, persistent=False)
>>>>>>> REPLACE
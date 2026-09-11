MECHANISM: Progressive pre-attention LayerNorm bias quotient

HYPOTHESIS: Reproducing the qualified 1,564-parameter design and removing one additional high-index `ln1.bias` direction will produce a 1,563-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reduce `ln1.bias` from seven to five orthonormal coordinates and `ln2.bias` from five to the qualified four coordinates; existing full-coordinate AdamW handling remains unchanged.

EVIDENCE: The first `ln1.bias` reduction combined with four-coordinate `ln2.bias` achieved 99.86% accuracy at 1,564 parameters, whereas further reducing `ln2.bias` failed at 98.62%; extending the successful independent `ln1` quotient is therefore the most informative next reduction.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 1))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln1_bias_basis", ln1_bias_basis, persistent=False)

        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)

        # Constant post-normalization shifts are absorbable by the
        # unrestricted fc1 bias. Retain five coordinates to test the next
        # progressive quotient beyond the qualified six-coordinate design.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 3))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 3)
        for col in range(cfg.d_model - 3):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_bias_basis[: col + 1, col] = 1.0 / scale
            ln2_bias_basis[col + 1, col] = -(col + 1) / scale
=======
        self.ln1 = nn.LayerNorm(cfg.d_model)

        # Constant pre-attention shifts are absorbable by the query and
        # output-projection biases, while key shifts cancel in the softmax.
        # Retain five coordinates, extending the qualified first reduction.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 3))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 3)
        for col in range(cfg.d_model - 3):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln1_bias_basis", ln1_bias_basis, persistent=False)

        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)

        # Constant post-normalization shifts are absorbable by the
        # unrestricted fc1 bias. Retain the qualified four coordinates.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 4))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 4)
        for col in range(cfg.d_model - 4):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_bias_basis[: col + 1, col] = 1.0 / scale
            ln2_bias_basis[col + 1, col] = -(col + 1) / scale
>>>>>>> REPLACE
MECHANISM: Progressive downstream-absorbed LayerNorm scale quotient

HYPOTHESIS: Reproducing the qualified 1,563-parameter design and fixing a second bias-free terminal `ln2.weight` coordinate at one will produce a 1,562-parameter model with at least 99% accuracy, because its effective scaling remains representable by the corresponding unrestricted `fc1` column.

INTENDED_EDIT: Use the qualified six-coordinate `ln1.bias` and four-coordinate `ln2.bias`, while reconstructing the final two `ln2` scales as constants instead of learned parameters.

EVIDENCE: The first terminal `ln2` scale quotient achieved 99.74% accuracy at 1,563 parameters; this extends that successful independent mechanism by one coordinate while avoiding the failed deeper `ln1.bias` and `ln2.bias` restrictions.

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
        self.register_buffer("ln2_bias_basis", ln2_bias_basis, persistent=False)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model)

        # Retain the qualified six-coordinate pre-attention bias quotient.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 2))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 2)
        for col in range(cfg.d_model - 2):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln1_bias_basis", ln1_bias_basis, persistent=False)

        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)

        # The compact bias basis leaves the terminal channels bias-free.
        # Their scales are absorbable by the corresponding unrestricted fc1
        # columns, so reconstruct the final two scales as fixed ones.
        self.ln2.weight = nn.Parameter(
            self.ln2.weight[:-2].detach().clone()
        )

        # Retain the qualified four-coordinate post-normalization bias.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 4))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 4)
        for col in range(cfg.d_model - 4):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_bias_basis[: col + 1, col] = 1.0 / scale
            ln2_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln2_bias_basis", ln2_bias_basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        ln2_bias = self.ln2_bias_basis @ self.ln2.bias
        normalized = F.layer_norm(
            x,
            self.ln2.normalized_shape,
            self.ln2.weight,
            ln2_bias,
            self.ln2.eps,
        )
=======
        ln2_bias = self.ln2_bias_basis @ self.ln2.bias
        ln2_weight = torch.cat(
            (self.ln2.weight, self.ln2.weight.new_ones(2))
        )
        normalized = F.layer_norm(
            x,
            self.ln2.normalized_shape,
            ln2_weight,
            ln2_bias,
            self.ln2.eps,
        )
>>>>>>> REPLACE
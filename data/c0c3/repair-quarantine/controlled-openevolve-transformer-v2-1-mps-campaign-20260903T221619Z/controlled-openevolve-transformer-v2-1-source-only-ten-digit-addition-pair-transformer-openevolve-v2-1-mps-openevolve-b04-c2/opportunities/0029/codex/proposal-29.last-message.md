MECHANISM: One-direction pre-attention LayerNorm bias quotient

HYPOTHESIS: Adding the verified zero-sum quotient of `ln1.bias` to the current 1,619-parameter model will reduce it to 1,618 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Represent `ln1.bias` in a seven-dimensional orthonormal zero-sum basis, reconstruct it in the forward pass, and preserve virtual full-coordinate AdamW updates during training.

EVIDENCE: Reference Design 2 used this exact extension and achieved 99.94% accuracy with 1,618 parameters; it is the smallest qualified design and directly improves the current verified model.

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model)

        # A uniform pre-attention shift is absorbable by the query bias and
        # output-projection bias; key shifts cancel from attention softmax.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 1))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln1_bias_basis", ln1_bias_basis, persistent=False)

        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        ln2_bias = self.ln2_bias_basis @ self.ln2.bias
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ln1_bias = self.ln1_bias_basis @ self.ln1.bias
        normalized = F.layer_norm(
            x,
            self.ln1.normalized_shape,
            self.ln1.weight,
            ln1_bias,
            self.ln1.eps,
        )
        x = x + self.attn(normalized)

        ln2_bias = self.ln2_bias_basis @ self.ln2.bias
>>>>>>> REPLACE

<<<<<<< SEARCH
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
=======
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
>>>>>>> REPLACE
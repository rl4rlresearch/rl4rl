MECHANISM: One-for-one exact-gauge substitution

HYPOTHESIS: Replacing the absorbable `ln1.bias` quotient with an eighth exactly function-null positional common-mode quotient will produce a 1,617-parameter model with at least 99% accuracy, because restoring the native pre-attention bias removes the optimization interaction present in failed compact-`ln1` reruns.

INTENDED_EDIT: Restore the full `ln1.bias`, compact the attention projection bias, extend positional compaction to the first two and final six rows, and update the quotient optimizer accordingly.

EVIDENCE: Reference Design 3 achieved 99.85% with native `ln1.bias`, compact projection bias, and six compact positional rows; the first positional extension reached 100%, while the failed eighth-row tests retained compact `ln1.bias`, motivating this equal-parameter substitution.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = nn.Linear(d_model, d_model)

        # A uniform projection-bias shift survives through the residual stream
        # but is erased by every subsequent LayerNorm, so store only its
        # zero-sum component.
        self.proj.bias = nn.Parameter(self.proj.bias.new_zeros(d_model - 1))
        proj_bias_basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            proj_bias_basis[: col + 1, col] = 1.0 / scale
            proj_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("proj_bias_basis", proj_bias_basis, persistent=False)
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        proj_bias = self.proj_bias_basis @ self.proj.bias
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model)

        # A uniform pre-attention shift is absorbable by the retained query
        # and output-projection biases; key shifts cancel from the softmax.
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 1))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln1_bias_basis", ln1_bias_basis, persistent=False)

        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
=======
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        x = x + self.attn(self.ln1(x))

        ln2_bias = self.ln2_bias_basis @ self.ln2.bias
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Preserve baseline initialization and RNG consumption, then remove
        # the LayerNorm-invariant common modes of rows zero, one, and the
        # final four positions, which have the shortest causal influence.
        self.compact_pos_count = 6
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-4:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-4].reshape(-1),
                )
            )
=======
        # Preserve baseline initialization and RNG consumption, then remove
        # the final-LayerNorm-invariant common modes of rows zero, one, and
        # the final six positions.
        self.compact_pos_count = 8
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-6:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-6].reshape(-1),
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
        )
=======
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                selected_update = torch.cat(
                    (full_update[:2], full_update[-4:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-4].reshape(-1),
                    )
                )
=======
                selected_update = torch.cat(
                    (full_update[:2], full_update[-6:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-6].reshape(-1),
                    )
                )
>>>>>>> REPLACE
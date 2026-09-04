MECHANISM: Three-row key-projection gauge quotient

HYPOTHESIS: Extending key compaction from one row to three will produce a 1,610-parameter model with at least 99% accuracy.

INTENDED_EDIT: Store the final three key-projection rows as independent seven-coordinate parameters, reconstruct them in one batched projection, and retain virtual eight-coordinate AdamW updates.

EVIDENCE: Reference Design 3 achieved 99.96% accuracy at 1,610 parameters and uses this exact extension of the current verified key-row quotient.

<<<<<<< SEARCH
        # One coordinate of a key row only produces a position-independent
        # key offset after LayerNorm and therefore cancels from the softmax.
        key_basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            key_basis[: col + 1, col] = 1.0 / scale
            key_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("key_basis", key_basis, persistent=False)

        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def compact_key_weight(self) -> None:
        with torch.no_grad():
            q_weight, k_weight, v_weight = self.qkv.weight.detach().chunk(3, dim=0)
            compact_key = k_weight[-1] @ self.key_basis
        self.q_weight = nn.Parameter(q_weight.clone())
        self.k_weight = nn.Parameter(k_weight[:-1].clone())
        self.k_compact = nn.Parameter(compact_key.clone())
        self.v_weight = nn.Parameter(v_weight.clone())
        del self.qkv
=======
        # For each compacted key row, one input common-mode coordinate only
        # produces a position-independent key offset and cancels in softmax.
        key_basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            key_basis[: col + 1, col] = 1.0 / scale
            key_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("key_basis", key_basis, persistent=False)

        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def compact_key_weights(self) -> None:
        with torch.no_grad():
            q_weight, k_weight, v_weight = self.qkv.weight.detach().chunk(3, dim=0)
            compact_key = k_weight[-3:] @ self.key_basis
        self.q_weight = nn.Parameter(q_weight.clone())
        self.k_weight = nn.Parameter(k_weight[:-3].clone())
        self.k_compact = nn.ParameterList(
            [nn.Parameter(row.clone()) for row in compact_key]
        )
        self.v_weight = nn.Parameter(v_weight.clone())
        del self.qkv
>>>>>>> REPLACE

<<<<<<< SEARCH
        key_main = F.linear(x, self.k_weight)
        key_last = F.linear(
            key_x @ self.key_basis, self.k_compact.unsqueeze(0)
        )
        k = torch.cat((key_main, key_last), dim=-1)
=======
        key_main = F.linear(x, self.k_weight)
        key_tail = F.linear(
            key_x @ self.key_basis,
            torch.stack(tuple(self.k_compact), dim=0),
        )
        k = torch.cat((key_main, key_tail), dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in self.blocks:
            blk.attn.compact_key_weight()
=======
        for blk in self.blocks:
            blk.attn.compact_key_weights()
>>>>>>> REPLACE

<<<<<<< SEARCH
    mlp_gauge_params = [
        (blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks
    ] + [
        (blk.attn.k_compact, blk.attn.key_basis) for blk in model.blocks
    ]
=======
    mlp_gauge_params = [
        (blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks
    ] + [
        (compact, blk.attn.key_basis)
        for blk in model.blocks
        for compact in blk.attn.k_compact
    ]
>>>>>>> REPLACE
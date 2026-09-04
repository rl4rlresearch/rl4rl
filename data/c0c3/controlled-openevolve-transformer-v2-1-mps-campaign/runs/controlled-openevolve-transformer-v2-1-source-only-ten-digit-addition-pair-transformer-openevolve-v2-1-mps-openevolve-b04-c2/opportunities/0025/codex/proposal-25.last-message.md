MECHANISM: Attention-output common-mode gauge fixing

HYPOTHESIS: Constraining the attention output-projection bias to the seven-dimensional zero-sum subspace will reduce the verified 1,613-parameter model to 1,612 parameters while retaining at least 99% accuracy, because its omitted uniform component passes unchanged through both residual additions and is canceled exactly by the downstream LayerNorms.

INTENDED_EDIT: Store the attention output-projection bias in the same orthonormal zero-sum basis used for the verified terminal MLP bias quotient, reconstruct it in the forward pass, and preserve virtual eight-coordinate AdamW moments during training.

EVIDENCE: The current bias-free-attention design achieved 99.84% accuracy at 1,613 parameters while already using a successful common-mode quotient for `fc2.bias`; applying that same exact residual-stream symmetry to `attn.proj.bias` is a distinct one-parameter reduction that avoids the positional, LayerNorm, and embedding reductions that previously failed.

<<<<<<< SEARCH
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.proj = nn.Linear(d_model, d_model)

        # A uniform output bias survives both residual additions unchanged
        # and is removed by each downstream LayerNorm, so retain only its
        # zero-sum coordinates.
        self.proj.bias = nn.Parameter(self.proj.bias.new_zeros(d_model - 1))
        proj_bias_basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            proj_bias_basis[: col + 1, col] = 1.0 / scale
            proj_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer(
            "proj_bias_basis", proj_bias_basis, persistent=False
        )
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
    mlp_gauge_params = [(blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks]
    pos_param = model.pos_emb.weight
    pos_basis = model.pos_basis
    compact_pos_count = model.compact_pos_count
    pos_compact_size = compact_pos_count * (model_cfg.d_model - 1)
    gauge_ids = {id(param) for param, _ in mlp_gauge_params}
    gauge_ids.add(id(pos_param))
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if id(param) not in gauge_ids],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    mlp_gauge_states = [
        {
            "step": 0,
            "exp_avg": basis.new_zeros(basis.size(0)),
            "exp_avg_sq": basis.new_zeros(basis.size(0)),
        }
        for _, basis in mlp_gauge_params
    ]
=======
    residual_gauge_params = [
        (blk.attn.proj.bias, blk.attn.proj_bias_basis)
        for blk in model.blocks
    ] + [
        (blk.mlp.fc2.bias, blk.mlp.bias_basis)
        for blk in model.blocks
    ]
    pos_param = model.pos_emb.weight
    pos_basis = model.pos_basis
    compact_pos_count = model.compact_pos_count
    pos_compact_size = compact_pos_count * (model_cfg.d_model - 1)
    gauge_ids = {id(param) for param, _ in residual_gauge_params}
    gauge_ids.add(id(pos_param))
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if id(param) not in gauge_ids],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    residual_gauge_states = [
        {
            "step": 0,
            "exp_avg": basis.new_zeros(basis.size(0)),
            "exp_avg_sq": basis.new_zeros(basis.size(0)),
        }
        for _, basis in residual_gauge_params
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
            for (param, basis), state in zip(mlp_gauge_params, mlp_gauge_states):
=======
            for (param, basis), state in zip(
                residual_gauge_params, residual_gauge_states
            ):
>>>>>>> REPLACE
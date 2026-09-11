MECHANISM: One-direction pre-attention LayerNorm bias quotient

HYPOTHESIS: Extending the verified 1,619-parameter design with a zero-sum quotient of `ln1.bias` will yield 1,618 parameters and at least 99% accuracy, because its removed uniform shift is absorbable by the retained query and projection biases.

INTENDED_EDIT: Apply the verified query-only attention, `ln2.bias`, and attention-projection quotients, then represent `ln1.bias` in a seven-dimensional zero-sum basis with reconstructed full-coordinate AdamW updates.

EVIDENCE: The independent `ln2.bias` and attention-projection quotients achieved 99.85% at 1,619 parameters, while removing a second `ln2` direction failed; this motivates testing one direction from the distinct, structurally redundant pre-attention bias.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Keep the original parameter slot and initialization RNG consumption,
        # but store only the query and value biases. A shared key bias cancels
        # from every row of the attention softmax.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # A shared key bias cancels from attention softmax. A shared value
        # bias passes unchanged through attention and is absorbable by the
        # retained output-projection bias, so store only the query bias.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)

        # A uniform projection-bias shift is erased by both subsequent
        # LayerNorms, so store only its zero-sum component.
        self.proj.bias = nn.Parameter(self.proj.bias.new_zeros(d_model - 1))
        proj_bias_basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            proj_bias_basis[: col + 1, col] = 1.0 / scale
            proj_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("proj_bias_basis", proj_bias_basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        q_bias, v_bias = self.qkv.bias.chunk(2)
        qkv_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
=======
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias
        qkv_bias = torch.cat(
            (q_bias, torch.zeros_like(q_bias), torch.zeros_like(q_bias))
        )
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        proj_bias = self.proj_bias_basis @ self.proj.bias
        y = F.linear(y, self.proj.weight, proj_bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
class Block(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
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
        self.ln2 = nn.LayerNorm(cfg.d_model)

        # A uniform component of this post-normalization bias is absorbable
        # by the unrestricted fc1 bias.
        self.ln2.bias = nn.Parameter(self.ln2.bias.new_zeros(cfg.d_model - 1))
        ln2_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln2_bias_basis[: col + 1, col] = 1.0 / scale
            ln2_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln2_bias_basis", ln2_bias_basis, persistent=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)

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
        normalized = F.layer_norm(
            x,
            self.ln2.normalized_shape,
            self.ln2.weight,
            ln2_bias,
            self.ln2.eps,
        )
        x = x + self.mlp(normalized)
        return x
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
    gauge_params = [
        pair
        for blk in model.blocks
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
        )
    ]
    pos_param = model.pos_emb.weight
    pos_basis = model.pos_basis
    compact_pos_count = model.compact_pos_count
    pos_compact_size = compact_pos_count * (model_cfg.d_model - 1)
    gauge_ids = {id(param) for param, _ in gauge_params}
    gauge_ids.add(id(pos_param))
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if id(param) not in gauge_ids],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_states = [
        {
            "step": 0,
            "exp_avg": basis.new_zeros(basis.size(0)),
            "exp_avg_sq": basis.new_zeros(basis.size(0)),
        }
        for _, basis in gauge_params
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Reconstruct full-coordinate AdamW moments for each quotient
        # parameter, then project updates into the compact coordinates.
        with torch.no_grad():
            beta1, beta2 = optimizer.defaults["betas"]
            eps = optimizer.defaults["eps"]

            for (param, basis), state in zip(mlp_gauge_params, mlp_gauge_states):
=======
        # Reconstruct full-coordinate AdamW moments for each quotient
        # parameter, then project updates into the compact coordinates.
        with torch.no_grad():
            beta1, beta2 = optimizer.defaults["betas"]
            eps = optimizer.defaults["eps"]

            for (param, basis), state in zip(gauge_params, gauge_states):
>>>>>>> REPLACE
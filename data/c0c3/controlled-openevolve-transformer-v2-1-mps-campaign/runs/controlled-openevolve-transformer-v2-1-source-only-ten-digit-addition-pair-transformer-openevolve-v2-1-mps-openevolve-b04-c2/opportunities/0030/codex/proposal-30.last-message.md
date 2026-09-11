MECHANISM: Batched four-row key-projection gauge quotient

HYPOTHESIS: Quotienting four key rows in one batched parameter will produce a 1,609-parameter model that completes training and achieves at least 99% accuracy; the three-row quotient achieved 99.96%, while the previous four-row attempt reported only a timeout, and batching removes its per-row execution overhead.

INTENDED_EDIT: Compact the final four key-projection rows into one seven-coordinate matrix, fuse the remaining query/key/value rows into one projection, and update the compact matrix with virtual eight-coordinate AdamW moments.

EVIDENCE: One-, two-, and three-row key quotients achieved 100%, 100%, and 99.96% accuracy respectively; the four-row extension had no reported accuracy failure because verification timed out, motivating an execution-efficient retry of the same incremental reduction.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Preserve constructor RNG consumption while omitting all QKV biases.
        # Key and value biases are functionally redundant, while a query
        # offset can be represented by the learned pre-attention LayerNorm
        # bias followed by the query projection.
        self.qkv.bias = None
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Preserve constructor RNG consumption while omitting all QKV biases.
        # Key and value biases are functionally redundant, while a query
        # offset can be represented by the learned pre-attention LayerNorm
        # bias followed by the query projection.
        self.qkv.bias = None

        # A common-mode input coordinate in each key row produces only a
        # position-independent key offset, which cancels inside softmax.
        self.compact_key_count = 4
        key_basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            key_basis[: col + 1, col] = 1.0 / scale
            key_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("key_basis", key_basis, persistent=False)

        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = F.linear(x, self.qkv.weight, None)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)

    def compact_key_weights(self) -> None:
        with torch.no_grad():
            q_weight, k_weight, v_weight = self.qkv.weight.detach().chunk(3, dim=0)
            main_weight = torch.cat(
                (
                    q_weight,
                    k_weight[: -self.compact_key_count],
                    v_weight,
                ),
                dim=0,
            )
            compact_key = (
                k_weight[-self.compact_key_count :] @ self.key_basis
            )
        self.main_weight = nn.Parameter(main_weight.clone())
        self.k_compact = nn.Parameter(compact_key.clone())
        del self.qkv

    def forward(self, x: torch.Tensor, key_x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        main = F.linear(x, self.main_weight)
        q, key_main, v = main.split(
            (
                d_model,
                d_model - self.compact_key_count,
                d_model,
            ),
            dim=-1,
        )
        key_tail = F.linear(
            key_x @ self.key_basis,
            self.k_compact,
        )
        k = torch.cat((key_main, key_tail), dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_input = self.ln1(x)
        key_input = F.layer_norm(
            x, (x.size(-1),), eps=self.ln1.eps
        )
        x = x + self.attn(attn_input, key_input)
        x = x + self.mlp(self.ln2(x))
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

        # Preserve baseline initialization and RNG consumption, then remove
=======
        self.apply(self._init_weights)
        for blk in self.blocks:
            blk.attn.compact_key_weights()

        # Preserve baseline initialization and RNG consumption, then remove
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
    pos_gauge_state = {
=======
    mlp_gauge_params = [
        (blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks
    ]
    key_gauge_params = [
        (blk.attn.k_compact, blk.attn.key_basis) for blk in model.blocks
    ]
    pos_param = model.pos_emb.weight
    pos_basis = model.pos_basis
    compact_pos_count = model.compact_pos_count
    pos_compact_size = compact_pos_count * (model_cfg.d_model - 1)
    gauge_ids = {id(param) for param, _ in mlp_gauge_params}
    gauge_ids.update(id(param) for param, _ in key_gauge_params)
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
    key_gauge_states = [
        {
            "step": 0,
            "exp_avg": param.new_zeros(param.size(0), basis.size(0)),
            "exp_avg_sq": param.new_zeros(param.size(0), basis.size(0)),
        }
        for param, basis in key_gauge_params
    ]
    pos_gauge_state = {
>>>>>>> REPLACE

<<<<<<< SEARCH
                param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                param.add_(quotient_update, alpha=-lr_now / bias_correction1)

            if pos_param.grad is not None:
=======
                param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                param.add_(quotient_update, alpha=-lr_now / bias_correction1)

            for (param, basis), state in zip(
                key_gauge_params, key_gauge_states
            ):
                if param.grad is None:
                    continue
                state["step"] += 1
                full_grad = param.grad @ basis.transpose(0, 1)
                state["exp_avg"].lerp_(full_grad, 1.0 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    full_grad, full_grad, value=1.0 - beta2
                )
                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denom = (
                    state["exp_avg_sq"].sqrt()
                    .div_(math.sqrt(bias_correction2))
                    .add_(eps)
                )
                quotient_update = (
                    state["exp_avg"] / denom
                ) @ basis
                param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                param.add_(
                    quotient_update,
                    alpha=-lr_now / bias_correction1,
                )

            if pos_param.grad is not None:
>>>>>>> REPLACE
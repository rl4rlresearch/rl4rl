MECHANISM: Incremental positional gauge fixing with full-coordinate AdamW

HYPOTHESIS: Extending the verified two-row positional quotient to row two will reduce the model from 1,633 to 1,632 parameters while retaining at least 99% accuracy, because two rows achieved 99.86% and quotienting all 23 rows only narrowly missed at 98.16%.

INTENDED_EDIT: Parameterize positional rows zero through two in the orthonormal zero-sum basis while preserving baseline initialization and reconstructed full-coordinate AdamW dynamics.

EVIDENCE: The verified two-row design reached 99.86% accuracy at 1,633 parameters; this supports the smallest incremental extension before attempting broader positional compression.

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)

        # Preserve the baseline initialization and its RNG consumption, then
        # remove rows zero through two's LayerNorm-invariant common modes.
        self.compact_pos_rows = 3
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            compact_pos = torch.cat(
                (
                    (
                        initialized_pos[: self.compact_pos_rows] @ pos_basis
                    ).reshape(-1),
                    initialized_pos[self.compact_pos_rows :].reshape(-1),
                )
            )
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        compact_size = self.compact_pos_rows * (self.cfg.d_model - 1)
        compact_rows = (
            self.pos_emb.weight[:compact_size].view(
                self.compact_pos_rows, self.cfg.d_model - 1
            )
            @ self.pos_basis.transpose(0, 1)
        )
        pos_weight = torch.cat(
            (
                compact_rows,
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_rows,
                    self.cfg.d_model,
                ),
            ),
            dim=0,
        )
        x = self.token_emb(idx) + F.embedding(pos, pos_weight)
        x = self.drop(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_params = [(blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks]
    gauge_ids = {id(param) for param, _ in gauge_params}
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
=======
    gauge_params = [(blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks]
    pos_param = model.pos_emb.weight
    pos_basis = model.pos_basis
    compact_pos_rows = model.compact_pos_rows
    pos_compact_size = compact_pos_rows * (model_cfg.d_model - 1)
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
    pos_gauge_state = {
        "step": 0,
        "exp_avg": pos_basis.new_zeros(model_cfg.max_seq_len * model_cfg.d_model),
        "exp_avg_sq": pos_basis.new_zeros(model_cfg.max_seq_len * model_cfg.d_model),
    }
>>>>>>> REPLACE

<<<<<<< SEARCH
                param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                param.add_(quotient_update, alpha=-lr_now / bias_correction1)

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
                param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                param.add_(quotient_update, alpha=-lr_now / bias_correction1)

            if pos_param.grad is not None:
                state = pos_gauge_state
                state["step"] += 1
                full_grad = torch.cat(
                    (
                        (
                            pos_param.grad[:pos_compact_size].view(
                                compact_pos_rows, model_cfg.d_model - 1
                            )
                            @ pos_basis.transpose(0, 1)
                        ).reshape(-1),
                        pos_param.grad[pos_compact_size:],
                    )
                )
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
                full_update = state["exp_avg"] / denom
                full_compact_size = compact_pos_rows * model_cfg.d_model
                compact_update = torch.cat(
                    (
                        (
                            full_update[:full_compact_size].view(
                                compact_pos_rows, model_cfg.d_model
                            )
                            @ pos_basis
                        ).reshape(-1),
                        full_update[full_compact_size:],
                    )
                )
                pos_param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                pos_param.add_(compact_update, alpha=-lr_now / bias_correction1)

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE
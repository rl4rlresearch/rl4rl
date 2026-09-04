MECHANISM: Single-row positional gauge fixing with full-coordinate AdamW

HYPOTHESIS: Extending the verified 1,635-parameter design by removing only one positional embedding common-mode coordinate will yield a 1,634-parameter model with at least 99% accuracy; the 1,612-parameter all-row quotient reached 98.16%, so limiting the same exact symmetry reduction to one row should greatly reduce its optimization disturbance.

INTENDED_EDIT: Apply the verified key-bias and terminal-MLP quotient reductions, then parameterize only positional row zero in a seven-dimensional orthonormal zero-sum basis while preserving baseline initialization RNG and reconstructed full-coordinate AdamW dynamics.

EVIDENCE: The terminal-bias quotient achieved 99.78% at 1,635 parameters, while quotienting all 23 positional rows produced a close 98.16% at 1,612; this tests the smallest conservative positional reduction using the already successful optimization treatment.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # A shared key bias cancels from every attention-softmax row. Retain
        # the original parameter slot while storing only query/value biases.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(2 * d_model))
        self.proj = nn.Linear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        q_bias, v_bias = self.qkv.bias.chunk(2)
        qkv_bias = torch.cat((q_bias, torch.zeros_like(q_bias), v_bias))
        qkv = F.linear(x, self.qkv.weight, qkv_bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

        # The following final LayerNorm removes a uniform residual-channel
        # shift, so parameterize this bias in an orthonormal zero-sum basis.
        self.fc2.bias = nn.Parameter(self.fc2.bias.new_zeros(d_model - 1))
        basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            basis[: col + 1, col] = 1.0 / scale
            basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("bias_basis", basis, persistent=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = self.bias_basis @ self.fc2.bias
        return self.drop(F.linear(hidden, self.fc2.weight, bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
=======
        self.apply(self._init_weights)

        # Preserve the baseline initialization and its RNG consumption, then
        # remove only row zero's LayerNorm-invariant common-mode coordinate.
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            compact_pos = torch.cat(
                (
                    pos_basis.transpose(0, 1) @ initialized_pos[0],
                    initialized_pos[1:].reshape(-1),
                )
            )
        self.pos_emb.weight = nn.Parameter(compact_pos)
        self.register_buffer("pos_basis", pos_basis, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        pos_weight = torch.cat(
            (
                (self.pos_basis @ self.pos_emb.weight[: self.cfg.d_model - 1]).unsqueeze(0),
                self.pos_emb.weight[self.cfg.d_model - 1 :].view(
                    self.cfg.max_seq_len - 1, self.cfg.d_model
                ),
            ),
            dim=0,
        )
        x = self.token_emb(idx) + F.embedding(pos, pos_weight)
        x = self.drop(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)

    mlp_gauge_params = [(blk.mlp.fc2.bias, blk.mlp.bias_basis) for blk in model.blocks]
    pos_param = model.pos_emb.weight
    pos_basis = model.pos_basis
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
        "step": 0,
        "exp_avg": pos_basis.new_zeros(model_cfg.max_seq_len * model_cfg.d_model),
        "exp_avg_sq": pos_basis.new_zeros(model_cfg.max_seq_len * model_cfg.d_model),
    }
>>>>>>> REPLACE

<<<<<<< SEARCH
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        _, loss = model(x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        # Use full-coordinate AdamW moments for each quotient parameter, then
        # project the updates back into its orthonormal compact coordinates.
        with torch.no_grad():
            beta1, beta2 = optimizer.defaults["betas"]
            eps = optimizer.defaults["eps"]

            for (param, basis), state in zip(mlp_gauge_params, mlp_gauge_states):
                if param.grad is None:
                    continue
                state["step"] += 1
                full_grad = basis @ param.grad
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
                quotient_update = basis.transpose(0, 1) @ (
                    state["exp_avg"] / denom
                )
                param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                param.add_(quotient_update, alpha=-lr_now / bias_correction1)

            if pos_param.grad is not None:
                state = pos_gauge_state
                state["step"] += 1
                full_grad = torch.cat(
                    (
                        pos_basis @ pos_param.grad[: model_cfg.d_model - 1],
                        pos_param.grad[model_cfg.d_model - 1 :],
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
                compact_update = torch.cat(
                    (
                        pos_basis.transpose(0, 1)
                        @ full_update[: model_cfg.d_model],
                        full_update[model_cfg.d_model :],
                    )
                )
                pos_param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                pos_param.add_(compact_update, alpha=-lr_now / bias_correction1)
>>>>>>> REPLACE
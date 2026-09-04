MECHANISM: Full-coordinate AdamW on quotient positional embeddings

HYPOTHESIS: Extending the verified 1,635-parameter design with row-wise zero-sum positional embeddings will remove one redundant parameter per sequence position while retaining at least 99% accuracy, provided initialization RNG and virtual eight-coordinate AdamW dynamics are preserved.

INTENDED_EDIT: Apply the verified key-bias and terminal-MLP quotient reductions, then store each positional embedding in a seven-dimensional orthonormal zero-sum basis and optimize both quotient parameter types through reconstructed full-coordinate AdamW moments.

EVIDENCE: The centered terminal-bias quotient reached 99.78% at 1,635 parameters, showing that orthonormal gauge removal succeeds when full-coordinate optimizer dynamics are retained; positional common-mode shifts are likewise canceled by every pre-norm path and the final LayerNorm.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # A shared key bias adds the same constant to every visible attention
        # logit and therefore cancels under the attention softmax.
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

        # The following final LayerNorm removes uniform residual-channel
        # shifts, so retain only an orthonormal zero-sum bias coordinate.
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
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)

        # A per-position uniform channel shift passes unchanged through the
        # residual stream but is invisible to every LayerNorm. Store each
        # positional vector in an orthonormal zero-sum basis.
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        self.pos_emb.weight = nn.Parameter(
            self.pos_emb.weight.new_zeros(cfg.max_seq_len, cfg.d_model - 1)
        )
        self.pos_emb.register_buffer(
            "zero_sum_basis", pos_basis, persistent=False
        )
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Embedding) and hasattr(module, "zero_sum_basis"):
            # Draw the original full tensor so all subsequent initialization
            # consumes exactly the baseline RNG stream, then discard only its
            # functionally invisible common-mode component.
            full_weight = module.weight.new_empty(
                module.num_embeddings, module.zero_sum_basis.size(0)
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full_weight @ module.zero_sum_basis)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        pos_coeff = self.pos_emb(pos)
        pos_full = pos_coeff @ self.pos_emb.zero_sum_basis.transpose(0, 1)
        x = self.token_emb(idx) + pos_full
        x = self.drop(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)

    gauge_params = [
        (model.pos_emb.weight, model.pos_emb.zero_sum_basis),
        *[
            (blk.mlp.fc2.bias, blk.mlp.bias_basis)
            for blk in model.blocks
        ],
    ]
    gauge_ids = {id(param) for param, _ in gauge_params}
    optimizer = torch.optim.AdamW(
        [param for param in model.parameters() if id(param) not in gauge_ids],
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_states = [
        {
            "step": 0,
            "exp_avg": basis.new_zeros(
                tuple(param.shape[:-1]) + (basis.size(0),)
            ),
            "exp_avg_sq": basis.new_zeros(
                tuple(param.shape[:-1]) + (basis.size(0),)
            ),
        }
        for param, basis in gauge_params
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
=======
        _, loss = model(x, y)
        model.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()

        # Run coordinate-wise AdamW in the original eight-dimensional spaces,
        # then project each update back into its orthonormal quotient basis.
        with torch.no_grad():
            beta1, beta2 = optimizer.defaults["betas"]
            eps = optimizer.defaults["eps"]
            basis_updates = [
                (param, basis, state)
                for (param, basis), state in zip(gauge_params, gauge_states)
                if param.grad is not None
            ]
            for param, basis, state in basis_updates:
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
                full_update = state["exp_avg"] / denom
                quotient_update = full_update @ basis
                param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                param.add_(quotient_update, alpha=-lr_now / bias_correction1)

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE
MECHANISM: Multi-column residual-output common-mode quotient

HYPOTHESIS: Extending the qualified 1,608-parameter design by quotienting the second attention output-projection column will yield 1,607 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified value, bias, and positional compactions, then parameterize the first two attention output-projection columns in a seven-dimensional zero-sum basis and optimize them with full-coordinate AdamW moments.

EVIDENCE: The 1,608-parameter reference achieved 99.90% after removing the function-null uniform component of its first projection column; the identical LayerNorm-erased direction exists independently in the second column.

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        if self.head_dim < 2:
            raise ValueError("value-basis compaction requires head dimension at least two")
        self.d_model = d_model

        # Align each head's first value-weight column with one basis vector.
        # The resulting zero subspace in the first head is further
        # triangularized through two second-column rotations.
        value_basis_rotations = []
        value_fixed_coordinates = []
        for head in range(n_head):
            value_basis_rotations.append((head, 0, 1, 0))
            value_fixed_coordinates.append((head, 0, 0))
            if self.head_dim >= 4:
                value_basis_rotations.append((head, 2, 3, 0))
                value_fixed_coordinates.append((head, 2, 0))
                value_basis_rotations.append((head, 1, 3, 0))
                value_fixed_coordinates.append((head, 1, 0))
                if head == 0:
                    value_basis_rotations.append((head, 0, 1, 1))
                    value_fixed_coordinates.append((head, 0, 1))
                    value_basis_rotations.append((head, 1, 2, 1))
                    value_fixed_coordinates.append((head, 1, 1))
        self.value_basis_rotations = tuple(value_basis_rotations)
        self.value_fixed_indices = tuple(
            sorted(
                (2 * d_model + head * self.head_dim + local) * d_model
                + input_column
                for head, local, input_column in value_fixed_coordinates
            )
        )

        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Shared key bias cancels from attention softmax, while shared value
        # bias is absorbable by the retained output-projection bias.
        self.qkv.bias = nn.Parameter(self.qkv.bias.new_zeros(d_model))
        self.proj = nn.Linear(d_model, d_model)

        # Subsequent LayerNorms erase uniform residual-channel shifts.
        self.proj.bias = nn.Parameter(self.proj.bias.new_zeros(d_model - 1))
        proj_bias_basis = torch.zeros(d_model, d_model - 1)
        for col in range(d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            proj_bias_basis[: col + 1, col] = 1.0 / scale
            proj_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("proj_bias_basis", proj_bias_basis, persistent=False)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
        self.register_buffer("mask", mask, persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.register_buffer("mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
=======
        self.register_buffer("mask", mask, persistent=False)

    def compact_value_basis(self) -> None:
        # Rotate value channels, counter-rotate their output columns, and omit
        # each coefficient made zero. Then remove the function-null uniform
        # output component from the first two projection columns.
        with torch.no_grad():
            qkv_weight = self.qkv.weight.detach().clone()
            proj_weight = self.proj.weight.detach().clone()

            for (
                head,
                first_local,
                second_local,
                input_column,
            ) in self.value_basis_rotations:
                first_value = (
                    2 * self.d_model + head * self.head_dim + first_local
                )
                second_value = (
                    2 * self.d_model + head * self.head_dim + second_local
                )
                first_column = head * self.head_dim + first_local
                second_column = head * self.head_dim + second_local

                a = qkv_weight[first_value, input_column]
                b = qkv_weight[second_value, input_column]
                norm = torch.hypot(a, b)
                cosine = b / norm
                sine = a / norm

                row0 = qkv_weight[first_value].clone()
                row1 = qkv_weight[second_value].clone()
                qkv_weight[first_value] = cosine * row0 - sine * row1
                qkv_weight[second_value] = sine * row0 + cosine * row1

                col0 = proj_weight[:, first_column].clone()
                col1 = proj_weight[:, second_column].clone()
                proj_weight[:, first_column] = cosine * col0 - sine * col1
                proj_weight[:, second_column] = sine * col0 + cosine * col1

            flat_weight = qkv_weight.reshape(-1)
            pieces = []
            start = 0
            for fixed_index in self.value_fixed_indices:
                pieces.append(flat_weight[start:fixed_index])
                start = fixed_index + 1
            pieces.append(flat_weight[start:])
            compact_weight = torch.cat(pieces)

            compact_proj_columns = (
                self.proj_bias_basis.transpose(0, 1) @ proj_weight[:, :2]
            )
            remaining_proj_weight = proj_weight[:, 2:].clone()

        self.qkv.weight = nn.Parameter(compact_weight)
        self.proj_compact_columns = nn.Parameter(compact_proj_columns)
        self.proj.weight = nn.Parameter(remaining_proj_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias
        qkv_bias = torch.cat(
            (q_bias, torch.zeros_like(q_bias), torch.zeros_like(q_bias))
        )
        weight_pieces = []
        compact_start = 0
        for removed, fixed_index in enumerate(self.value_fixed_indices):
            compact_index = fixed_index - removed
            weight_pieces.append(self.qkv.weight[compact_start:compact_index])
            weight_pieces.append(self.qkv.weight.new_zeros(1))
            compact_start = compact_index
        weight_pieces.append(self.qkv.weight[compact_start:])
        qkv_weight = torch.cat(weight_pieces).view(
            3 * self.d_model, self.d_model
        )
        qkv = F.linear(x, qkv_weight, qkv_bias)
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
        compact_columns = self.proj_bias_basis @ self.proj_compact_columns
        proj_weight = torch.cat((compact_columns, self.proj.weight), dim=1)
        y = F.linear(y, proj_weight, proj_bias)
        y = self.resid_drop(y)
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

        # The final LayerNorm removes a uniform residual-channel shift.
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
        self.ln1.bias = nn.Parameter(self.ln1.bias.new_zeros(cfg.d_model - 1))
        ln1_bias_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            ln1_bias_basis[: col + 1, col] = 1.0 / scale
            ln1_bias_basis[col + 1, col] = -(col + 1) / scale
        self.register_buffer("ln1_bias_basis", ln1_bias_basis, persistent=False)

        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln2 = nn.LayerNorm(cfg.d_model)
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
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)
        for block in self.blocks:
            block.attn.compact_value_basis()

        # Remove LayerNorm-invariant common modes from the first two and final
        # five positional rows while preserving baseline RNG consumption.
        self.compact_pos_count = 7
        pos_basis = torch.zeros(cfg.d_model, cfg.d_model - 1)
        for col in range(cfg.d_model - 1):
            scale = math.sqrt((col + 1) * (col + 2))
            pos_basis[: col + 1, col] = 1.0 / scale
            pos_basis[col + 1, col] = -(col + 1) / scale
        with torch.no_grad():
            initialized_pos = self.pos_emb.weight.detach()
            selected_pos = torch.cat(
                (initialized_pos[:2], initialized_pos[-5:]), dim=0
            )
            compact_pos = torch.cat(
                (
                    (selected_pos @ pos_basis).reshape(-1),
                    initialized_pos[2:-5].reshape(-1),
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
        compact_size = self.compact_pos_count * (self.cfg.d_model - 1)
        compact_rows = (
            self.pos_emb.weight[:compact_size].view(
                self.compact_pos_count, self.cfg.d_model - 1
            )
            @ self.pos_basis.transpose(0, 1)
        )
        pos_weight = torch.cat(
            (
                compact_rows[:2],
                self.pos_emb.weight[compact_size:].view(
                    self.cfg.max_seq_len - self.compact_pos_count,
                    self.cfg.d_model,
                ),
                compact_rows[2:],
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

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)

    gauge_params = [
        pair
        for blk in model.blocks
        for pair in (
            (blk.mlp.fc2.bias, blk.mlp.bias_basis),
            (blk.ln1.bias, blk.ln1_bias_basis),
            (blk.ln2.bias, blk.ln2_bias_basis),
            (blk.attn.proj.bias, blk.attn.proj_bias_basis),
            (blk.attn.proj_compact_columns, blk.attn.proj_bias_basis),
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
            "exp_avg": basis.new_zeros(
                (basis.size(0), *param.shape[1:])
            ),
            "exp_avg_sq": basis.new_zeros(
                (basis.size(0), *param.shape[1:])
            ),
        }
        for param, basis in gauge_params
    ]
    pos_gauge_state = {
        "step": 0,
        "exp_avg": pos_basis.new_zeros(model_cfg.max_seq_len * model_cfg.d_model),
        "exp_avg_sq": pos_basis.new_zeros(model_cfg.max_seq_len * model_cfg.d_model),
    }

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
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

        # Maintain AdamW moments in the original full coordinates for each
        # quotient parameter, then project updates into compact coordinates.
        with torch.no_grad():
            beta1, beta2 = optimizer.defaults["betas"]
            eps = optimizer.defaults["eps"]

            for (param, basis), state in zip(gauge_params, gauge_states):
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
                selected_full_grad = (
                    pos_param.grad[:pos_compact_size].view(
                        compact_pos_count, model_cfg.d_model - 1
                    )
                    @ pos_basis.transpose(0, 1)
                )
                full_grad = torch.cat(
                    (
                        selected_full_grad[:2].reshape(-1),
                        pos_param.grad[pos_compact_size:],
                        selected_full_grad[2:].reshape(-1),
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
                full_update = (
                    state["exp_avg"] / denom
                ).view(model_cfg.max_seq_len, model_cfg.d_model)
                selected_update = torch.cat(
                    (full_update[:2], full_update[-5:]), dim=0
                )
                compact_update = torch.cat(
                    (
                        (selected_update @ pos_basis).reshape(-1),
                        full_update[2:-5].reshape(-1),
                    )
                )
                pos_param.mul_(1.0 - lr_now * train_cfg.weight_decay)
                pos_param.add_(compact_update, alpha=-lr_now / bias_correction1)

        if (step % train_cfg.eval_interval == 0) or (step == train_cfg.train_steps - 1):
>>>>>>> REPLACE
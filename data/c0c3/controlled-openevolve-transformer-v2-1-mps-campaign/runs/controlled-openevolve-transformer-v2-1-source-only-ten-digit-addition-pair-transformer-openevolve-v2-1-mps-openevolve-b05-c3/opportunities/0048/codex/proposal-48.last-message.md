MECHANISM: Third attention-output weight-column shift quotient

HYPOTHESIS: Extending the verified two-column projection gauge to a third input column will reduce the model to 1,593 parameters while retaining at least 99% accuracy, because the omitted coefficient contributes only an input-dependent channel-common residual shift removed by downstream LayerNorms.

INTENDED_EDIT: Adopt the qualified positional, query-bias-sharing, fixed-MLP-bias, and key-row reductions, then gauge-fix three attention-output projection columns while preserving full-space initialization, AdamW dynamics, weight decay, and gradient clipping.

EVIDENCE: The two-column projection quotient achieved 99.95% accuracy at 1,594 parameters after the one-column quotient achieved 99.93% at 1,595; the third column applies the same exact symmetry, unlike the unrelated 1,596-parameter reductions that failed.

<<<<<<< SEARCH
    vocab_size: int


class CausalSelfAttention(nn.Module):
=======
    vocab_size: int


class GaugeFixedEmbedding(nn.Embedding):
    """Embedding vectors represented modulo a shared channel shift."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim - 1))

    def full_weight(self) -> torch.Tensor:
        zero = self.weight.new_zeros(self.num_embeddings, 1)
        return torch.cat((self.weight, zero), dim=-1)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return F.embedding(
            idx,
            self.full_weight(),
            self.padding_idx,
            self.max_norm,
            self.norm_type,
            self.scale_grad_by_freq,
            self.sparse,
        )


class GaugeFixedKeyLinear(nn.Linear):
    """QKV projection with one key-row coefficient fixed."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.missing_index = d_model * d_model + d_model - 1
        self.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 1))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_index],
                self.weight.new_zeros(1),
                self.weight[self.missing_index :],
            )
        )
        return flat.view(3 * self.d_model, self.d_model)


class GaugeFixedProjectionLinear(nn.Linear):
    """Attention projection with three common-output shifts fixed."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.missing_count = 3
        self.missing_start = (d_model - 1) * d_model
        self.weight = nn.Parameter(
            torch.empty(d_model * d_model - self.missing_count)
        )

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(self.missing_count),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_model)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.attn_drop = nn.Dropout(dropout)
=======
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.qkv = GaugeFixedKeyLinear(d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 7))
        self.proj = GaugeFixedProjectionLinear(d_model)
        self.attn_drop = nn.Dropout(dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        bsz, seqlen, d_model = x.shape
        independent_query_bias = torch.cat(
            (self.qkv.bias, self.proj.bias[-1:])
        )
        query_bias = torch.cat(
            (
                independent_query_bias,
                independent_query_bias.mean().unsqueeze(0),
            )
        )
        bias = torch.cat(
            (query_bias, self.qkv.bias.new_zeros(2 * d_model + 5))
        )
        qkv = F.linear(x, self.qkv.full_weight(), bias)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = F.linear(y, self.proj.full_weight(), self.proj.bias)
        y = self.resid_drop(y)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.fc1.bias, self.fc1.bias.new_zeros(1)))
        hidden = F.linear(x, self.fc1.weight, bias)
        return self.drop(self.fc2(F.gelu(hidden)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedEmbedding(cfg.max_seq_len, cfg.d_model)
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
        if isinstance(module, GaugeFixedEmbedding):
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full[:, :-1] - full[:, -1:])
        elif isinstance(module, GaugeFixedKeyLinear):
            d_model = module.d_model
            full = module.weight.new_empty(3 * d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[d_model, -1].clone()
                full[d_model, :-1].sub_(omitted)
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_index],
                            flat[module.missing_index + 1 :],
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedProjectionLinear):
            d_model = module.d_model
            count = module.missing_count
            full = module.weight.new_empty(d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1, :count].clone()
                full[:, :count].sub_(omitted)
                full[-1, :count].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + count :],
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


def save_json(path: Path, obj: Dict) -> None:
=======
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


class GaugeOptimizer:
    """Full-space AdamW dynamics for all quotient parameters."""

    def __init__(
        self,
        model: torch.nn.Module,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        self.position_parameter = model.pos_emb.weight
        self.key_modules = [
            (block.attn.qkv, block.ln1.weight)
            for block in model.blocks
        ]
        self.projection_modules = [
            block.attn.proj for block in model.blocks
        ]
        self.shared_parameters = [
            block.attn.proj.bias for block in model.blocks
        ]
        self.parameters = [self.position_parameter]
        self.parameters.extend(
            module.weight for module, _ in self.key_modules
        )
        self.parameters.extend(
            module.weight for module in self.projection_modules
        )
        self.parameters.extend(self.shared_parameters)
        self.param_groups = [
            {
                "lr": lr,
                "weight_decay": weight_decay,
                "betas": betas,
                "eps": eps,
            }
        ]
        self.state = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            if set_to_none:
                parameter.grad = None
            else:
                parameter.grad.zero_()

    def _adam_update(
        self,
        key,
        gradient: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        group = self.param_groups[0]
        beta1, beta2 = group["betas"]
        if key not in self.state:
            self.state[key] = {
                "step": 0,
                "exp_avg": torch.zeros_like(gradient),
                "exp_avg_sq": torch.zeros_like(gradient),
            }
        state = self.state[key]
        state["step"] += 1
        state["exp_avg"].mul_(beta1).add_(
            gradient, alpha=1.0 - beta1
        )
        state["exp_avg_sq"].mul_(beta2).addcmul_(
            gradient, gradient, value=1.0 - beta2
        )

        step = state["step"]
        bias_correction1 = 1.0 - beta1 ** step
        bias_correction2 = 1.0 - beta2 ** step
        denominator = state["exp_avg_sq"].sqrt().div_(
            math.sqrt(bias_correction2)
        ).add_(group["eps"])

        value.mul_(1.0 - group["lr"] * group["weight_decay"])
        value.addcdiv_(
            state["exp_avg"],
            denominator,
            value=-group["lr"] / bias_correction1,
        )
        return value

    @staticmethod
    def shared_gradients(
        parameter: torch.nn.Parameter,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        gradient = parameter.grad
        projection_gradient = gradient.clone()
        projection_gradient[-1] = -gradient[:-1].sum()
        query_gradient = gradient[-1] - projection_gradient[-1]
        return projection_gradient, query_gradient

    @staticmethod
    def key_gradient(
        module: torch.nn.Module,
        ln_scale: torch.nn.Parameter,
    ) -> torch.Tensor:
        d_model = module.d_model
        missing_index = module.missing_index
        parameter = module.weight
        full_gradient = torch.cat(
            (
                parameter.grad[:missing_index],
                parameter.grad.new_zeros(1),
                parameter.grad[missing_index:],
            )
        ).view(3 * d_model, d_model)
        gamma = ln_scale.detach()
        full_gradient[d_model, -1] = -gamma[-1] * (
            full_gradient[d_model, :-1] / gamma[:-1]
        ).sum()
        return full_gradient

    @staticmethod
    def projection_gradient(module: torch.nn.Module) -> torch.Tensor:
        d_model = module.d_model
        count = module.missing_count
        start = module.missing_start
        parameter = module.weight
        full_gradient = torch.cat(
            (
                parameter.grad[:start],
                parameter.grad.new_zeros(count),
                parameter.grad[start:],
            )
        ).view(d_model, d_model)
        full_gradient[-1, :count] = -full_gradient[
            :-1, :count
        ].sum(dim=0)
        return full_gradient

    @torch.no_grad()
    def step(self) -> None:
        position = self.position_parameter
        if position.grad is not None:
            full_gradient = torch.cat(
                (
                    position.grad,
                    -position.grad.sum(dim=-1, keepdim=True),
                ),
                dim=-1,
            )
            full_value = torch.cat(
                (
                    position,
                    position.new_zeros(*position.shape[:-1], 1),
                ),
                dim=-1,
            )
            full_value = self._adam_update(
                (id(position), "position"),
                full_gradient,
                full_value,
            )
            position.copy_(
                full_value[..., :-1] - full_value[..., -1:]
            )

        for module, ln_scale in self.key_modules:
            parameter = module.weight
            if parameter.grad is None:
                continue
            full_gradient = self.key_gradient(module, ln_scale)
            full_value = self._adam_update(
                (id(parameter), "key"),
                full_gradient,
                module.full_weight(),
            )
            d_model = module.d_model
            gamma = ln_scale.detach()
            key_row = full_value[d_model]
            omitted = key_row[-1].clone()
            key_row[:-1].sub_(
                omitted * gamma[-1] / gamma[:-1]
            )
            key_row[-1].zero_()
            flat = full_value.reshape(-1)
            parameter.copy_(
                torch.cat(
                    (
                        flat[: module.missing_index],
                        flat[module.missing_index + 1 :],
                    )
                )
            )

        for module in self.projection_modules:
            parameter = module.weight
            if parameter.grad is None:
                continue
            full_gradient = self.projection_gradient(module)
            full_value = self._adam_update(
                (id(parameter), "projection_weight"),
                full_gradient,
                module.full_weight(),
            )
            count = module.missing_count
            omitted = full_value[-1, :count].clone()
            full_value[:, :count].sub_(omitted)
            full_value[-1, :count].zero_()
            flat = full_value.reshape(-1)
            parameter.copy_(
                torch.cat(
                    (
                        flat[: module.missing_start],
                        flat[module.missing_start + count :],
                    )
                )
            )

        for parameter in self.shared_parameters:
            if parameter.grad is None:
                continue
            projection_gradient, query_gradient = (
                self.shared_gradients(parameter)
            )
            projection_value = self._adam_update(
                (id(parameter), "projection_bias"),
                projection_gradient,
                parameter.clone(),
            )
            query_value = self._adam_update(
                (id(parameter), "query_bias"),
                query_gradient,
                parameter[-1].clone(),
            )
            parameter[:-1].copy_(
                projection_value[:-1]
                - projection_value[-1]
                + query_value
            )
            parameter[-1].copy_(query_value)

    @torch.no_grad()
    def clip_grad_norm(
        self,
        model: torch.nn.Module,
        max_norm: float,
    ) -> None:
        total_sq = torch.zeros(
            (),
            device=self.position_parameter.device,
            dtype=torch.float32,
        )
        key_by_id = {
            id(module.weight): (module, ln_scale)
            for module, ln_scale in self.key_modules
        }
        projection_by_id = {
            id(module.weight): module
            for module in self.projection_modules
        }
        shared_ids = {
            id(parameter) for parameter in self.shared_parameters
        }

        for parameter in model.parameters():
            if parameter.grad is None:
                continue
            parameter_id = id(parameter)
            gradient = parameter.grad.detach().float()

            if parameter_id in shared_ids:
                projection_gradient, query_gradient = (
                    self.shared_gradients(parameter)
                )
                total_sq.add_(
                    projection_gradient.detach().float().square().sum()
                )
                total_sq.add_(
                    query_gradient.detach().float().square()
                )
            else:
                total_sq.add_(gradient.square().sum())

            if parameter is self.position_parameter:
                total_sq.add_(
                    gradient.sum(dim=-1).square().sum()
                )

            if parameter_id in key_by_id:
                module, ln_scale = key_by_id[parameter_id]
                full_gradient = self.key_gradient(module, ln_scale)
                total_sq.add_(
                    full_gradient[
                        module.d_model, -1
                    ].detach().float().square()
                )

            if parameter_id in projection_by_id:
                module = projection_by_id[parameter_id]
                full_gradient = self.projection_gradient(module)
                total_sq.add_(
                    full_gradient[
                        -1, : module.missing_count
                    ].detach().float().square().sum()
                )

        coefficient = (
            max_norm / (total_sq.sqrt() + 1e-6)
        ).clamp(max=1.0)
        for parameter in model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(
                    coefficient.to(dtype=parameter.grad.dtype)
                )


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
    gauge_optimizer = GaugeOptimizer(
        model,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    excluded = {
        id(parameter) for parameter in gauge_optimizer.parameters
    }
    regular_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in excluded
    ]
    optimizer = torch.optim.AdamW(
        regular_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    all_optimizers = [optimizer, gauge_optimizer]

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
>>>>>>> REPLACE

<<<<<<< SEARCH
        lr_now = cosine_lr(step, train_cfg.train_steps, train_cfg.lr, train_cfg.warmup_steps, train_cfg.min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        lr_now = cosine_lr(step, train_cfg.train_steps, train_cfg.lr, train_cfg.warmup_steps, train_cfg.min_lr_ratio)
        for current_optimizer in all_optimizers:
            for pg in current_optimizer.param_groups:
                pg["lr"] = lr_now

        _, loss = model(x, y)
        for current_optimizer in all_optimizers:
            current_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            gauge_optimizer.clip_grad_norm(
                model,
                train_cfg.grad_clip,
            )
        gauge_optimizer.step()
        optimizer.step()
>>>>>>> REPLACE
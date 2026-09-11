MECHANISM: Learned relative-lag attention routing

HYPOTHESIS: Replacing independent absolute positional residuals with two learned head-specific relative-lag tables will produce 1,476 parameters and retain at least 99% accuracy, because addition decoding can learn stationary attention routes while preserving the full-rank token representations shown to be load-bearing.

INTENDED_EDIT: Remove the 161-parameter absolute position embedding, add a gauge-fixed 44-parameter causal lag-bias table, and retain the qualified two-column projection and final-MLP output-bias quotients.

EVIDENCE: The qualified 1,593-parameter design reached 99.96%, whereas rank-seven token factorization collapsed to 5.06%; this motivates preserving token identity capacity while challenging the shared assumption that every sequence position needs an independent learned residual vector.

<<<<<<< SEARCH
class GaugeFixedProjectionLinear(nn.Linear):
    """Attention projection with one common-output-shift coordinate fixed."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.missing_index = (d_model - 1) * d_model
        self.weight = nn.Parameter(torch.empty(d_model * d_model - 1))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_index],
                self.weight.new_zeros(1),
                self.weight[self.missing_index :],
            )
        )
        return flat.view(self.d_model, self.d_model)
=======
class GaugeFixedProjectionLinear(nn.Linear):
    """Attention projection with two common-output-shift coordinates fixed."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.missing_start = (d_model - 1) * d_model
        self.weight = nn.Parameter(torch.empty(d_model * d_model - 2))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(2),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.proj = GaugeFixedProjectionLinear(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
=======
        self.proj = GaugeFixedProjectionLinear(d_model)
        self.attn_drop = nn.Dropout(dropout)
        self.resid_drop = nn.Dropout(dropout)

        # Each head learns a stationary causal routing preference. The omitted
        # final lag fixes the softmax-invariant common shift of each table.
        self.relative_bias = nn.Parameter(
            torch.zeros(n_head, max_seq_len - 1)
        )

        mask = torch.tril(torch.ones(max_seq_len, max_seq_len, dtype=torch.bool))
>>>>>>> REPLACE

<<<<<<< SEARCH
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
=======
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        positions = torch.arange(seqlen, device=x.device)
        lag = (positions[:, None] - positions[None, :]).clamp_min(0)
        lag_bias = torch.cat(
            (
                self.relative_bias,
                self.relative_bias.new_zeros(self.n_head, 1),
            ),
            dim=-1,
        )
        att = att + lag_bias[:, lag].unsqueeze(0)
        causal = self.mask[:seqlen, :seqlen]
        att = att.masked_fill(~causal, float("-inf"))
        att = F.softmax(att, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.fc1.bias, self.fc1.bias.new_zeros(1)))
        hidden = F.linear(x, self.fc1.weight, bias)
        return self.drop(self.fc2(F.gelu(hidden)))
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 1))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.fc1.bias, self.fc1.bias.new_zeros(1)))
        hidden = F.linear(x, self.fc1.weight, bias)
        output_bias = torch.cat(
            (self.fc2.bias, self.fc2.bias.new_zeros(1))
        )
        output = F.linear(F.gelu(hidden), self.fc2.weight, output_bias)
        return self.drop(output)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = GaugeFixedEmbedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
=======
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedProjectionLinear):
            d_model = module.d_model
            full = module.weight.new_empty(d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1, 0].clone()
                full[:, 0].sub_(omitted)
                full[-1, 0].zero_()
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
=======
        elif isinstance(module, GaugeFixedProjectionLinear):
            d_model = module.d_model
            full = module.weight.new_empty(d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1, :2].clone()
                full[:, :2].sub_(omitted)
                full[-1, :2].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 2 :],
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        pos = torch.arange(0, seqlen, device=idx.device).unsqueeze(0)
        x = self.token_emb(idx) + self.pos_emb(pos)
        x = self.drop(x)
=======
        x = self.drop(self.token_emb(idx))
>>>>>>> REPLACE

<<<<<<< SEARCH
def full_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    missing_index = (d_model - 1) * d_model
    full_grad = torch.cat(
        (
            parameter.grad[:missing_index],
            parameter.grad.new_zeros(1),
            parameter.grad[missing_index:],
        )
    ).view(d_model, d_model)
    full_grad[-1, 0] = -full_grad[:-1, 0].sum()
    return full_grad
=======
def full_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    missing_start = (d_model - 1) * d_model
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(2),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_model)
    full_grad[-1, :2] = -full_grad[:-1, :2].sum(dim=0)
    return full_grad
>>>>>>> REPLACE

<<<<<<< SEARCH
        d_model = self.module.d_model
        missing_index = self.module.missing_index
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                full_grad = full_projection_gradient(parameter, d_model)
=======
        d_model = self.module.d_model
        missing_start = self.module.missing_start
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                full_grad = full_projection_gradient(parameter, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                column = full_value[:, 0]
                omitted = column[-1].clone()
                column.sub_(omitted)
                column[-1].zero_()

                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_index],
                            flat[missing_index + 1 :],
                        )
                    )
                )
=======
                omitted = full_value[-1, :2].clone()
                full_value[:, :2].sub_(omitted)
                full_value[-1, :2].zero_()

                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 2 :],
                        )
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
@torch.no_grad()
def clip_grad_norm_with_gauges(
    model: torch.nn.Module,
    position_parameter: torch.nn.Parameter,
    key_gauges: List[
        Tuple[torch.nn.Parameter, torch.nn.Parameter, int]
    ],
    projection_gauges: List[Tuple[torch.nn.Parameter, int]],
    shared_parameters: List[torch.nn.Parameter],
    max_norm: float,
) -> None:
    total_sq = torch.zeros(
        (), device=position_parameter.device, dtype=torch.float32
    )
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()

        if any(
            parameter is shared_parameter
            for shared_parameter in shared_parameters
        ):
            projection_gradient, query_gradient = (
                shared_query_projection_gradients(parameter)
            )
            total_sq.add_(
                projection_gradient.detach().float().square().sum()
            )
            total_sq.add_(
                query_gradient.detach().float().square()
            )
        else:
            total_sq.add_(grad.square().sum())

        if parameter is position_parameter:
            total_sq.add_(grad.sum(dim=-1).square().sum())

        for key_parameter, ln_scale, d_model in key_gauges:
            if parameter is key_parameter:
                full_grad = full_key_gradient(
                    key_parameter, ln_scale, d_model
                )
                total_sq.add_(
                    full_grad[d_model, -1].float().square()
                )

        for projection_parameter, d_model in projection_gauges:
            if parameter is projection_parameter:
                full_grad = full_projection_gradient(
                    projection_parameter, d_model
                )
                total_sq.add_(
                    full_grad[-1, 0].float().square()
                )

    coefficient = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(
                coefficient.to(dtype=parameter.grad.dtype)
            )
=======
@torch.no_grad()
def clip_grad_norm_with_gauges(
    model: torch.nn.Module,
    last_coordinate_gauges: List[torch.nn.Parameter],
    key_gauges: List[
        Tuple[torch.nn.Parameter, torch.nn.Parameter, int]
    ],
    projection_gauges: List[Tuple[torch.nn.Parameter, int]],
    shared_parameters: List[torch.nn.Parameter],
    max_norm: float,
) -> None:
    reference_parameter = next(model.parameters())
    total_sq = torch.zeros(
        (), device=reference_parameter.device, dtype=torch.float32
    )
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()

        if any(
            parameter is shared_parameter
            for shared_parameter in shared_parameters
        ):
            projection_gradient, query_gradient = (
                shared_query_projection_gradients(parameter)
            )
            total_sq.add_(
                projection_gradient.detach().float().square().sum()
            )
            total_sq.add_(
                query_gradient.detach().float().square()
            )
        else:
            total_sq.add_(grad.square().sum())

        if any(
            parameter is gauge_parameter
            for gauge_parameter in last_coordinate_gauges
        ):
            total_sq.add_(
                grad.sum(dim=-1).square().sum()
            )

        for key_parameter, ln_scale, d_model in key_gauges:
            if parameter is key_parameter:
                full_grad = full_key_gradient(
                    key_parameter, ln_scale, d_model
                )
                total_sq.add_(
                    full_grad[d_model, -1].float().square()
                )

        for projection_parameter, d_model in projection_gauges:
            if parameter is projection_parameter:
                full_grad = full_projection_gradient(
                    projection_parameter, d_model
                )
                total_sq.add_(
                    full_grad[-1, :2].float().square().sum()
                )

    coefficient = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(
                coefficient.to(dtype=parameter.grad.dtype)
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    position_parameter = model.pos_emb.weight
    key_gauge_modules = [
        (block.attn.qkv, block.ln1.weight)
        for block in model.blocks
    ]
    projection_gauge_modules = [
        block.attn.proj for block in model.blocks
    ]
    shared_query_projection_parameters = [
        block.attn.proj.bias for block in model.blocks
    ]
    excluded = {id(position_parameter)}
    excluded.update(id(module.weight) for module, _ in key_gauge_modules)
    excluded.update(id(module.weight) for module in projection_gauge_modules)
    excluded.update(
        id(parameter)
        for parameter in shared_query_projection_parameters
    )
=======
    output_bias_gauge_parameters = [
        block.mlp.fc2.bias for block in model.blocks
    ]
    relative_bias_gauge_parameters = [
        block.attn.relative_bias for block in model.blocks
    ]
    last_coordinate_gauge_parameters = [
        *output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
    ]
    key_gauge_modules = [
        (block.attn.qkv, block.ln1.weight)
        for block in model.blocks
    ]
    projection_gauge_modules = [
        block.attn.proj for block in model.blocks
    ]
    shared_query_projection_parameters = [
        block.attn.proj.bias for block in model.blocks
    ]
    excluded = {
        id(parameter)
        for parameter in last_coordinate_gauge_parameters
    }
    excluded.update(id(module.weight) for module, _ in key_gauge_modules)
    excluded.update(id(module.weight) for module in projection_gauge_modules)
    excluded.update(
        id(parameter)
        for parameter in shared_query_projection_parameters
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
    position_optimizer = GaugeAdamW(
        position_parameter,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    key_optimizers = [
=======
    last_coordinate_optimizers = [
        GaugeAdamW(
            parameter,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for parameter in last_coordinate_gauge_parameters
    ]
    key_optimizers = [
>>>>>>> REPLACE

<<<<<<< SEARCH
    all_optimizers = [
        optimizer,
        position_optimizer,
        *key_optimizers,
        *projection_optimizers,
        *shared_optimizers,
    ]
=======
    all_optimizers = [
        optimizer,
        *last_coordinate_optimizers,
        *key_optimizers,
        *projection_optimizers,
        *shared_optimizers,
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
            clip_grad_norm_with_gauges(
                model,
                position_parameter,
                key_gauges,
                projection_gauges,
                shared_query_projection_parameters,
                train_cfg.grad_clip,
            )
=======
            clip_grad_norm_with_gauges(
                model,
                last_coordinate_gauge_parameters,
                key_gauges,
                projection_gauges,
                shared_query_projection_parameters,
                train_cfg.grad_clip,
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for shared_optimizer in shared_optimizers:
            shared_optimizer.step()
        position_optimizer.step()
        optimizer.step()
=======
        for shared_optimizer in shared_optimizers:
            shared_optimizer.step()
        for gauge_optimizer in last_coordinate_optimizers:
            gauge_optimizer.step()
        optimizer.step()
>>>>>>> REPLACE
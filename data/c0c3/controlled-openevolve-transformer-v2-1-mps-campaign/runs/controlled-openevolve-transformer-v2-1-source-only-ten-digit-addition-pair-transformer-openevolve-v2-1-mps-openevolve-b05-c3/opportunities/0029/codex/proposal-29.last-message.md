MECHANISM: Gauge-fixed attention output bias under downstream LayerNorm shift invariance

HYPOTHESIS: Quotienting the shared-channel shift of each attention output-projection bias will reduce the qualified 1600-parameter design to 1599 parameters while retaining at least 99% accuracy, because the omitted scalar changes the residual stream only by a channel-constant offset that both downstream LayerNorms remove, and gauge-aware AdamW preserves full-space optimization dynamics.

INTENDED_EDIT: Adopt the qualified four-coordinate query bias and fixed trailing `fc1` bias, represent each eight-coordinate attention projection bias with seven learned coordinates and one fixed-zero gauge coordinate, and optimize all gauge parameters with full-space-equivalent AdamW and gradient clipping.

EVIDENCE: The four-query-bias design with one fixed `fc1` bias achieved 99.91% at 1600 parameters, while several direct 1599-parameter capacity ablations failed; the successful positional quotient further shows that exact invariances are promising when full-space AdamW moments and clipping norms are preserved.

<<<<<<< SEARCH
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Five query coordinates are
        # learned; the remaining query coordinates and key/value biases are zero.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 3))
        self.proj = nn.Linear(d_model, d_model)
=======
        # Construct with baseline shapes first so subsequent modules retain the
        # proven initialization RNG stream. Four query coordinates are learned;
        # the remaining query coordinates and key/value biases are zero.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
        self.proj = nn.Linear(d_model, d_model)
        # A shared shift of this bias adds only a channel-constant residual
        # offset, which the downstream LayerNorms remove exactly.
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 3))
        )
=======
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 4))
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        y = self.proj(y)
        y = self.resid_drop(y)
=======
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, d_model)
        proj_bias = torch.cat(
            (self.proj.bias, self.proj.bias.new_zeros(1))
        )
        y = F.linear(y, self.proj.weight, proj_bias)
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
        # Preserve the baseline constructor RNG stream, then fix the qualified
        # trailing hidden-unit bias coordinate at zero.
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
class GaugeAdamW(torch.optim.Optimizer):
    """AdamW on an embedding quotient with one virtual gauge coordinate."""
=======
class GaugeAdamW(torch.optim.Optimizer):
    """AdamW on a last-coordinate-zero shift quotient."""
>>>>>>> REPLACE

<<<<<<< SEARCH
@torch.no_grad()
def clip_grad_norm_with_gauge(
    model: torch.nn.Module,
    gauge_parameter: torch.nn.Parameter,
    max_norm: float,
) -> None:
    total_sq = torch.zeros(
        (), device=gauge_parameter.device, dtype=torch.float32
    )
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        total_sq.add_(grad.square().sum())
        if parameter is gauge_parameter:
            total_sq.add_(grad.sum(dim=-1).square().sum())

    coefficient = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(
                coefficient.to(dtype=parameter.grad.dtype)
            )
=======
@torch.no_grad()
def clip_grad_norm_with_gauge(
    model: torch.nn.Module,
    gauge_parameters: List[torch.nn.Parameter],
    max_norm: float,
) -> None:
    total_sq = torch.zeros(
        (), device=gauge_parameters[0].device, dtype=torch.float32
    )
    gauge_ids = {id(parameter) for parameter in gauge_parameters}
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        total_sq.add_(grad.square().sum())
        if id(parameter) in gauge_ids:
            total_sq.add_(grad.sum(dim=-1).square().sum())

    coefficient = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
    for parameter in model.parameters():
        if parameter.grad is not None:
            parameter.grad.mul_(
                coefficient.to(dtype=parameter.grad.dtype)
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameter = model.pos_emb.weight
    regular_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter is not gauge_parameter
    ]
    optimizer = torch.optim.AdamW(
        regular_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_optimizer = GaugeAdamW(
        gauge_parameter,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
=======
    gauge_parameters = [
        model.pos_emb.weight,
        *(block.attn.proj.bias for block in model.blocks),
    ]
    gauge_ids = {id(parameter) for parameter in gauge_parameters}
    regular_parameters = [
        parameter
        for parameter in model.parameters()
        if id(parameter) not in gauge_ids
    ]
    optimizer = torch.optim.AdamW(
        regular_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_optimizers = [
        GaugeAdamW(
            parameter,
            lr=train_cfg.lr,
            weight_decay=train_cfg.weight_decay,
        )
        for parameter in gauge_parameters
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        for current_optimizer in (optimizer, gauge_optimizer):
            for pg in current_optimizer.param_groups:
                pg["lr"] = lr_now
=======
        for current_optimizer in (optimizer, *gauge_optimizers):
            for pg in current_optimizer.param_groups:
                pg["lr"] = lr_now
>>>>>>> REPLACE

<<<<<<< SEARCH
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_gauge(
                model, gauge_parameter, train_cfg.grad_clip
            )
        optimizer.step()
        gauge_optimizer.step()
=======
        optimizer.zero_grad(set_to_none=True)
        for gauge_optimizer in gauge_optimizers:
            gauge_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_gauge(
                model, gauge_parameters, train_cfg.grad_clip
            )
        optimizer.step()
        for gauge_optimizer in gauge_optimizers:
            gauge_optimizer.step()
>>>>>>> REPLACE
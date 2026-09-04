MECHANISM: Joint token-position translation quotient with full-space-equivalent AdamW

HYPOTHESIS: Removing all seven token-position translation redundancies from the verified 1600-parameter design will produce 1593 parameters and retain at least 99% accuracy because the patch preserves its initialized function, full-space gradients, AdamW moments, weight decay, and clipping norm.

INTENDED_EDIT: Adopt the qualified positional and four-query-bias reductions plus the single fixed `fc1` bias, then fix the first position’s seven represented coordinates at zero while compensating through the tied token embedding.

EVIDENCE: The 1600-parameter design achieved 99.91%, and the positional quotient succeeded when full-space optimizer dynamics were preserved; unlike the failed embedding-only shift quotient, this joint translation leaves every transformer input exactly unchanged and changes output logits only by a vocabulary-wide common offset.

<<<<<<< SEARCH
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class CausalSelfAttention(nn.Module):
=======
@dataclass
class ModelConfig:
    n_layer: int
    d_model: int
    n_head: int
    d_ff: int
    dropout: float
    max_seq_len: int
    vocab_size: int


class JointGaugePositionEmbedding(nn.Embedding):
    """Position embeddings modulo row shifts and token-position translations."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume the constructor RNG used by the original full embedding.
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty((num_embeddings - 1) * (embedding_dim - 1))
        )
        self.register_buffer(
            "initial_shift",
            torch.zeros(embedding_dim - 1),
            persistent=False,
        )

    def full_weight(self) -> torch.Tensor:
        first = self.weight.new_zeros(1, self.embedding_dim - 1)
        reduced = torch.cat(
            (
                first,
                self.weight.view(
                    self.num_embeddings - 1, self.embedding_dim - 1
                ),
            ),
            dim=0,
        )
        return torch.cat(
            (reduced, reduced.new_zeros(self.num_embeddings, 1)),
            dim=-1,
        )

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


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Key bias is softmax-invariant,
        # while value bias is a constant absorbed by the output bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
=======
        # Construct the baseline layer first to preserve the qualified RNG
        # stream. Only the first four query-bias coordinates remain learned.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat((self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model)))
=======
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 4))
        )
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
        # Preserve the baseline constructor stream while fixing only the
        # qualified trailing hidden-unit bias coordinate at zero.
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
=======
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = JointGaugePositionEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
=======
        self.apply(self._init_weights)
        # Complete the joint gauge transformation after the tied output layer
        # performs the baseline's final token-embedding initialization draw.
        with torch.no_grad():
            self.token_emb.weight[:, :-1].add_(
                self.pos_emb.initial_shift
            )
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
        if isinstance(module, JointGaugePositionEmbedding):
            # Draw the original full tensor, select the last-channel-zero row
            # gauge, then move the first position vector into the tied token
            # matrix through the exact joint translation symmetry.
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            row_gauge = full[:, :-1] - full[:, -1:]
            shift = row_gauge[0].clone()
            joint_gauge = row_gauge - shift
            with torch.no_grad():
                module.weight.copy_(joint_gauge[1:].reshape(-1))
                module.initial_shift.copy_(shift)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    return min_lr + (base_lr - min_lr) * cosine


def save_json(path: Path, obj: Dict) -> None:
=======
    return min_lr + (base_lr - min_lr) * cosine


@torch.no_grad()
def full_position_gradient(
    token_parameter: torch.nn.Parameter,
    position_parameter: torch.nn.Parameter,
    num_positions: int,
    d_model: int,
) -> torch.Tensor:
    """Recover the gradient in the original full position-embedding space."""
    first = position_parameter.grad.new_zeros(
        num_positions, d_model - 1
    )
    first[1:].copy_(
        position_parameter.grad.view(num_positions - 1, d_model - 1)
    )
    # Joint translation invariance determines the omitted first-row gradient.
    first[0].copy_(
        token_parameter.grad[:, :-1].sum(dim=0)
        - first[1:].sum(dim=0)
    )
    # Per-position common-channel invariance determines the last coordinate.
    return torch.cat(
        (first, -first.sum(dim=-1, keepdim=True)),
        dim=-1,
    )


class JointGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW followed by an exact joint gauge projection."""

    def __init__(
        self,
        position_parameter: torch.nn.Parameter,
        token_parameter: torch.nn.Parameter,
        num_positions: int,
        d_model: int,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(
            [position_parameter],
            dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps),
        )
        self.token_parameter = token_parameter
        self.num_positions = num_positions
        self.d_model = d_model

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if (
                    parameter.grad is None
                    or self.token_parameter.grad is None
                ):
                    continue

                full_grad = full_position_gradient(
                    self.token_parameter,
                    parameter,
                    self.num_positions,
                    self.d_model,
                )
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(full_grad)
                    state["exp_avg_sq"] = torch.zeros_like(full_grad)

                state["step"] += 1
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                exp_avg.mul_(beta1).add_(full_grad, alpha=1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    full_grad, full_grad, value=1.0 - beta2
                )

                bias_correction1 = 1.0 - beta1 ** state["step"]
                bias_correction2 = 1.0 - beta2 ** state["step"]
                denom = exp_avg_sq.sqrt().div_(
                    math.sqrt(bias_correction2)
                ).add_(group["eps"])

                first = parameter.new_zeros(
                    self.num_positions, self.d_model - 1
                )
                first[1:].copy_(
                    parameter.view(
                        self.num_positions - 1, self.d_model - 1
                    )
                )
                full_value = torch.cat(
                    (
                        first,
                        first.new_zeros(self.num_positions, 1),
                    ),
                    dim=-1,
                )
                full_value.mul_(
                    1.0 - group["lr"] * group["weight_decay"]
                )
                full_value.addcdiv_(
                    exp_avg,
                    denom,
                    value=-group["lr"] / bias_correction1,
                )

                # Restore both gauges without changing the represented model.
                full_value = full_value - full_value[:, -1:]
                shift = full_value[0, :-1].clone()
                full_value[:, :-1].sub_(shift)
                self.token_parameter[:, :-1].add_(shift)
                parameter.copy_(
                    full_value[1:, :-1].reshape_as(parameter)
                )

        return loss


@torch.no_grad()
def clip_grad_norm_with_joint_gauge(
    model: torch.nn.Module,
    token_parameter: torch.nn.Parameter,
    position_parameter: torch.nn.Parameter,
    num_positions: int,
    d_model: int,
    max_norm: float,
) -> None:
    full_position_grad = full_position_gradient(
        token_parameter,
        position_parameter,
        num_positions,
        d_model,
    )
    total_sq = torch.zeros(
        (), device=position_parameter.device, dtype=torch.float32
    )
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        if parameter is position_parameter:
            grad = full_position_grad.detach().float()
        else:
            grad = parameter.grad.detach().float()
        total_sq.add_(grad.square().sum())

    coefficient = (max_norm / (total_sq.sqrt() + 1e-6)).clamp(max=1.0)
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
=======
    model = TinyDecoderLM(model_cfg).to(device)
    params = count_parameters(model)
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
    gauge_optimizer = JointGaugeAdamW(
        gauge_parameter,
        model.token_emb.weight,
        model_cfg.max_seq_len,
        model_cfg.d_model,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
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
        for current_optimizer in (optimizer, gauge_optimizer):
            for pg in current_optimizer.param_groups:
                pg["lr"] = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_joint_gauge(
                model,
                model.token_emb.weight,
                gauge_parameter,
                model_cfg.max_seq_len,
                model_cfg.d_model,
                train_cfg.grad_clip,
            )
        optimizer.step()
        gauge_optimizer.step()
>>>>>>> REPLACE
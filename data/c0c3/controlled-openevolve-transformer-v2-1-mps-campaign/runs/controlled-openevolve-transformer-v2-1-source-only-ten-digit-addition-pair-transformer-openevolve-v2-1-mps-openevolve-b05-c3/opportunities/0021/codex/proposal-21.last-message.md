MECHANISM: Global tied-embedding shift quotient with gauge-aware AdamW

HYPOTHESIS: Gauge-fixing the single all-entries shift symmetry of the tied token-embedding/output matrix will reduce the verified 1601-parameter design to 1600 parameters while retaining at least 99% accuracy, because the shift is invisible to every LayerNorm-mediated residual block and adds only a common offset to all output logits.

INTENDED_EDIT: Adopt the qualified seven-coordinate positional embedding and four-coordinate query bias, then represent the tied vocabulary matrix with one global scalar omitted while preserving baseline initialization draws, full-space AdamW moments, and gradient clipping.

EVIDENCE: The four-query-bias positional-quotient design achieved 99.96% at 1601 parameters, and the positional quotient succeeded after full-space optimizer dynamics were preserved; the rank-seven tied-interface attempt failed at 0.01%, motivating removal of only an exact global shift redundancy rather than an embedding rank direction.

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


class GaugeFixedTokenEmbedding(nn.Embedding):
    """Tied vocabulary matrix modulo one global all-entries shift."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume the constructor RNG used by the original full embedding.
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings * embedding_dim - 1)
        )

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat((self.weight, self.weight.new_zeros(1)))
        return flat.view(self.num_embeddings, self.embedding_dim)

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

    @torch.no_grad()
    def initialize_from_full(self, full: torch.Tensor) -> None:
        gauged = full - full[-1, -1]
        self.weight.copy_(gauged.reshape(-1)[:-1])


class GaugeFixedEmbedding(nn.Embedding):
    """Position vectors represented modulo per-vector channel shifts."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume the constructor RNG used by the original full embedding.
        super().__init__(num_embeddings, embedding_dim)
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim - 1)
        )

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


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Key bias is softmax-invariant,
        # while value bias is a constant absorbed by the output bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
=======
        # Construct the baseline layer first to preserve its RNG stream. Only
        # the first four query-bias coordinates remain learned; key and value
        # biases and the other query coordinates are fixed at zero.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat((self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model)))
        qkv = F.linear(x, self.qkv.weight, bias)
=======
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 4))
        )
        qkv = F.linear(x, self.qkv.weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Weight tying with input embeddings.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
=======
        self.cfg = cfg
        self.token_emb = GaugeFixedTokenEmbedding(
            cfg.vocab_size, cfg.d_model
        )
        self.pos_emb = GaugeFixedEmbedding(
            cfg.max_seq_len, cfg.d_model
        )
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.d_model)

        # Keep the shared parameter alias and the original constructor RNG
        # consumption; forward reconstructs its gauge-fixed matrix.
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.lm_head._uses_global_gauge_weight = True
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
        if isinstance(module, GaugeFixedTokenEmbedding):
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            module.initialize_from_full(full)
        elif isinstance(module, GaugeFixedEmbedding):
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full[:, :-1] - full[:, -1:])
        elif (
            isinstance(module, nn.Linear)
            and getattr(module, "_uses_global_gauge_weight", False)
        ):
            # The baseline initializes the tied matrix again when visiting the
            # head, so draw that same full tensor before selecting its gauge.
            full = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            gauged = full - full[-1, -1]
            with torch.no_grad():
                module.weight.copy_(gauged.reshape(-1)[:-1])
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x)
        logits = F.linear(x, self.token_emb.full_weight())
>>>>>>> REPLACE

<<<<<<< SEARCH
def save_json(path: Path, obj: Dict) -> None:
=======
class GaugeAdamW(torch.optim.Optimizer):
    """AdamW with the omitted coordinates restored in optimizer state."""

    def __init__(
        self,
        parameter: torch.nn.Parameter,
        lr: float,
        weight_decay: float,
        global_shift: bool = False,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(
            [parameter],
            dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps),
        )
        self.global_shift = global_shift

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                reduced_grad = parameter.grad
                if self.global_shift:
                    reduced_flat = reduced_grad.reshape(-1)
                    full_grad = torch.cat(
                        (
                            reduced_flat,
                            -reduced_flat.sum().reshape(1),
                        )
                    )
                else:
                    full_grad = torch.cat(
                        (
                            reduced_grad,
                            -reduced_grad.sum(dim=-1, keepdim=True),
                        ),
                        dim=-1,
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

                step = state["step"]
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step
                denom = exp_avg_sq.sqrt().div_(
                    math.sqrt(bias_correction2)
                ).add_(group["eps"])

                if self.global_shift:
                    full_value = torch.cat(
                        (parameter.reshape(-1), parameter.new_zeros(1))
                    )
                else:
                    full_value = torch.cat(
                        (
                            parameter,
                            parameter.new_zeros(
                                *parameter.shape[:-1], 1
                            ),
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

                if self.global_shift:
                    parameter.copy_(
                        (
                            full_value[:-1] - full_value[-1]
                        ).view_as(parameter)
                    )
                else:
                    parameter.copy_(
                        full_value[..., :-1] - full_value[..., -1:]
                    )

        return loss


@torch.no_grad()
def clip_grad_norm_with_gauges(
    model: torch.nn.Module,
    position_parameter: torch.nn.Parameter,
    token_parameter: torch.nn.Parameter,
    max_norm: float,
) -> None:
    total_sq = torch.zeros(
        (), device=position_parameter.device, dtype=torch.float32
    )
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        total_sq.add_(grad.square().sum())
        if parameter is position_parameter:
            total_sq.add_(grad.sum(dim=-1).square().sum())
        elif parameter is token_parameter:
            total_sq.add_(grad.sum().square())

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
    position_parameter = model.pos_emb.weight
    token_parameter = model.token_emb.weight
    regular_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter is not position_parameter
        and parameter is not token_parameter
    ]
    optimizer = torch.optim.AdamW(
        regular_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    position_optimizer = GaugeAdamW(
        position_parameter,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    token_optimizer = GaugeAdamW(
        token_parameter,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
        global_shift=True,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        optimizer.step()
=======
        for current_optimizer in (
            optimizer,
            position_optimizer,
            token_optimizer,
        ):
            for pg in current_optimizer.param_groups:
                pg["lr"] = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        position_optimizer.zero_grad(set_to_none=True)
        token_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_gauges(
                model,
                position_parameter,
                token_parameter,
                train_cfg.grad_clip,
            )
        optimizer.step()
        position_optimizer.step()
        token_optimizer.step()
>>>>>>> REPLACE
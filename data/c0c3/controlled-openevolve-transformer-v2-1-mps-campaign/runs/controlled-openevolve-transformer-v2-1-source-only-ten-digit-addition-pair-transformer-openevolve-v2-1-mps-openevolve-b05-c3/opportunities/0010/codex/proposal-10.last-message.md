MECHANISM: Gauge-quotient positional embeddings with AdamW-equivalent updates

HYPOTHESIS: Representing each positional vector with seven coordinates modulo its LayerNorm-invisible common-channel shift will remove `max_seq_len` parameters from the qualified 1627-parameter model while retaining at least 99% accuracy, provided initialization, gradient clipping, and AdamW updates preserve the original eight-coordinate quotient dynamics.

INTENDED_EDIT: Replace the full positional table with a last-coordinate-zero gauge representation, preserve the proven constructor and initialization RNG streams, and train it with a gauge-aware AdamW optimizer that reconstructs the omitted gradient and optimizer state.

EVIDENCE: The 1627-parameter query-bias design reached 99.88%, while an earlier positional gauge edit failed at 4.5%; the key-bias experiments showed that an analytically redundant change recovered from 86.15% to 99.86% when the original RNG stream was preserved. This patch additionally preserves AdamW and clipping dynamics, directly addressing a second load-bearing assumption of the failed gauge parameterization.

<<<<<<< SEARCH
    vocab_size: int


class CausalSelfAttention(nn.Module):
=======
    vocab_size: int


class GaugeFixedEmbedding(nn.Embedding):
    """Embedding vectors represented modulo a shared channel shift."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume exactly the constructor RNG used by the original embedding.
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


class CausalSelfAttention(nn.Module):
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
            # Draw the original full tensor so every later initialization keeps
            # the qualified RNG stream, then choose the equivalent gauge whose
            # final coordinate is zero.
            full = module.weight.new_empty(
                module.num_embeddings, module.embedding_dim
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                module.weight.copy_(full[:, :-1] - full[:, -1:])
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
def save_json(path: Path, obj: Dict) -> None:
=======
class GaugeAdamW(torch.optim.Optimizer):
    """AdamW on an embedding quotient with one virtual gauge coordinate."""

    def __init__(
        self,
        parameter: torch.nn.Parameter,
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(
            [parameter],
            dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps),
        )

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

                full_value = torch.cat(
                    (
                        parameter,
                        parameter.new_zeros(*parameter.shape[:-1], 1),
                    ),
                    dim=-1,
                )
                full_value.mul_(1.0 - group["lr"] * group["weight_decay"])
                full_value.addcdiv_(
                    exp_avg,
                    denom,
                    value=-group["lr"] / bias_correction1,
                )
                parameter.copy_(
                    full_value[..., :-1] - full_value[..., -1:]
                )

        return loss


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
            # Shift invariance implies the omitted full gradient coordinate.
            total_sq.add_(grad.sum(dim=-1).square().sum())

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

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
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
    gauge_optimizer = GaugeAdamW(
        gauge_parameter,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )

    sampler = TrainBatchSampler(train_cfg.batch_size, train_cfg.seed + 1337, reserved_hashes)
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
        for current_optimizer in (optimizer, gauge_optimizer):
            for pg in current_optimizer.param_groups:
                pg["lr"] = lr_now

        _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        gauge_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if train_cfg.grad_clip > 0:
            clip_grad_norm_with_gauge(
                model, gauge_parameter, train_cfg.grad_clip
            )
        optimizer.step()
        gauge_optimizer.step()
>>>>>>> REPLACE
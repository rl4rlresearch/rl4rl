MECHANISM: Shared-threshold pair in the MLP hidden bias

HYPOTHESIS: Tying hidden-bias coordinates 9 and 10 while retaining the verified fixed-zero coordinate 11 will produce 1599 parameters and at least 99% accuracy, because it preserves a trainable threshold for both units whereas the prior 1599 design forced both trailing thresholds to zero and narrowly missed at 98.57%.

INTENDED_EDIT: Adopt the qualified positional gauge, four-coordinate query bias, and gauge-aware optimization, then represent the 12-coordinate `fc1` bias with ten learned values, using the final learned value for coordinates 9 and 10 and fixing coordinate 11 at zero.

EVIDENCE: The 1600-parameter design with only coordinate 11 fixed achieved 99.91%, while additionally fixing coordinate 10 reached 98.57%; replacing that second zero constraint with a shared learned threshold is a lower-distortion test in the same nearly qualified parameter group.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class GaugeFixedEmbedding(nn.Embedding):
    """Embedding vectors represented modulo a shared channel shift."""

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Consume the constructor RNG used by the original full embedding.
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
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Key bias is softmax-invariant,
        # while value bias is a constant absorbed by the output bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
=======
        # Preserve the baseline constructor RNG stream. Only four query-bias
        # coordinates are learned; all remaining query and key/value biases
        # are fixed at zero.
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
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
        # Preserve the baseline constructor RNG stream, then remove two stored
        # coordinates. Hidden units 9 and 10 share one learned threshold, while
        # the already-qualified final coordinate remains fixed at zero.
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc1.bias = nn.Parameter(torch.empty(d_ff - 2))
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (
                self.fc1.bias,
                self.fc1.bias[-1:],
                self.fc1.bias.new_zeros(1),
            )
        )
        hidden = F.linear(x, self.fc1.weight, bias)
        return self.drop(self.fc2(F.gelu(hidden)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.pos_emb = nn.Embedding(cfg.max_seq_len, cfg.d_model)
=======
        self.pos_emb = GaugeFixedEmbedding(cfg.max_seq_len, cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, GaugeFixedEmbedding):
            # Draw the original full tensor to preserve the qualified RNG
            # stream, then select the equivalent last-coordinate-zero gauge.
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
=======
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
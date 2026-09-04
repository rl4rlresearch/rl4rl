MECHANISM: LayerNorm-hyperplane quotient for one MLP input row

HYPOTHESIS: Fixing the qualified trailing `fc1` bias and quotienting unit 10’s eight effective normalized-input coefficients to seven will produce 1599 learned parameters and retain at least 99% accuracy.

INTENDED_EDIT: Restore the qualified four-coordinate query bias, retain eleven learned `fc1` thresholds, and represent unit 10 directly on the zero-mean LayerNorm coordinates with one omitted gauge coefficient and full-space-equivalent AdamW moments and clipping.

EVIDENCE: The 1600-parameter design with one fixed `fc1` bias achieved 99.91%, while removing a second threshold missed at 98.57%; the unverified unit-10 weight-quotient proposal therefore remains the most informative way to remove one scalar without sacrificing that unit’s learned threshold.

<<<<<<< SEARCH
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Five query coordinates are
        # learned; the remaining query coordinates and key/value biases are zero.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 3))
=======
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Four query coordinates are
        # learned; the remaining query coordinates and key/value biases are zero.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
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
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
class GaugeFixedFC1(nn.Module):
    """MLP input map with one LayerNorm-hyperplane gauge removed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        if d_ff < 2:
            raise ValueError("d_ff must be at least two")

        # Consume the constructor RNG used by the original full Linear.
        baseline = nn.Linear(d_model, d_ff)
        del baseline

        self.d_model = d_model
        self.d_ff = d_ff
        self.gauge_index = d_ff - 2
        self.regular_weight = nn.Parameter(
            torch.empty(d_ff - 1, d_model)
        )
        self.gauge_weight = nn.Parameter(torch.empty(d_model - 1))
        # Coordinates 0 through d_ff-2 remain learned; the final bias is zero.
        self.bias = nn.Parameter(torch.empty(d_ff - 1))

    def forward(
        self, x: torch.Tensor, layer_norm: nn.LayerNorm
    ) -> torch.Tensor:
        normalized = F.layer_norm(
            x,
            (self.d_model,),
            weight=None,
            bias=None,
            eps=layer_norm.eps,
        )
        affine = layer_norm(x)

        regular_bias = torch.cat(
            (
                self.bias[: self.gauge_index],
                self.bias.new_zeros(1),
            )
        )
        regular = F.linear(
            affine, self.regular_weight, regular_bias
        )
        gauge = F.linear(
            normalized,
            self.gauge_weight.unsqueeze(0),
            self.bias[self.gauge_index : self.gauge_index + 1],
        )
        return torch.cat(
            (
                regular[..., : self.gauge_index],
                gauge,
                regular[..., self.gauge_index :],
            ),
            dim=-1,
        )


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = GaugeFixedFC1(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, layer_norm: nn.LayerNorm
    ) -> torch.Tensor:
        hidden = self.fc1(x, layer_norm)
        return self.drop(self.fc2(F.gelu(hidden)))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(x, self.ln2)
        return x
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
        elif isinstance(module, GaugeFixedFC1):
            full = module.regular_weight.new_empty(
                module.d_ff, module.d_model
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            gauge_index = module.gauge_index
            with torch.no_grad():
                module.regular_weight.copy_(
                    torch.cat(
                        (
                            full[:gauge_index],
                            full[gauge_index + 1 :],
                        ),
                        dim=0,
                    )
                )
                module.gauge_weight.copy_(
                    full[gauge_index, :-1] - full[gauge_index, -1]
                )
                module.bias.zero_()
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
class GaugeAdamW(torch.optim.Optimizer):
    """AdamW on last-coordinate-zero quotient parameters."""

    def __init__(
        self,
        parameters: List[torch.nn.Parameter],
        lr: float,
        weight_decay: float,
        betas: Tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ):
        super().__init__(
            parameters,
            dict(lr=lr, weight_decay=weight_decay, betas=betas, eps=eps),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
def clip_grad_norm_with_gauge(
    model: torch.nn.Module,
    gauge_parameters: List[torch.nn.Parameter],
    max_norm: float,
) -> None:
    total_sq = torch.zeros(
        (), device=gauge_parameters[0].device, dtype=torch.float32
    )
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        total_sq.add_(grad.square().sum())
        if any(parameter is gauge for gauge in gauge_parameters):
            total_sq.add_(grad.sum(dim=-1).square().sum())
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
    params = count_parameters(model)
    gauge_parameters = [model.pos_emb.weight]
    gauge_parameters.extend(
        block.mlp.fc1.gauge_weight for block in model.blocks
    )
    regular_parameters = [
        parameter
        for parameter in model.parameters()
        if not any(parameter is gauge for gauge in gauge_parameters)
    ]
    optimizer = torch.optim.AdamW(
        regular_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
    gauge_optimizer = GaugeAdamW(
        gauge_parameters,
        lr=train_cfg.lr,
        weight_decay=train_cfg.weight_decay,
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
            clip_grad_norm_with_gauge(
                model, gauge_parameter, train_cfg.grad_clip
            )
=======
            clip_grad_norm_with_gauge(
                model, gauge_parameters, train_cfg.grad_clip
            )
>>>>>>> REPLACE
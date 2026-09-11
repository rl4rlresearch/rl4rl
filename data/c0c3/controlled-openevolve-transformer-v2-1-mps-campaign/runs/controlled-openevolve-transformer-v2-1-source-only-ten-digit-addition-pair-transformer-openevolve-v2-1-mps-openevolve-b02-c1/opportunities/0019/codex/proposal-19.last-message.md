MECHANISM: Optimizer-preserving residual gauge quotient

HYPOTHESIS: Eliminating the 12 LayerNorm-invisible common-mode `fc2` weights while initializing from full-width weight differences and emulating the omitted row’s AdamW moments will produce a 1,564-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reduce `fc2` to seven residual coordinates, preserve the successful full-width observable initialization, and train it with quotient-aware AdamW updates and gradient clipping.

EVIDENCE: Direct and orthonormal seven-coordinate `fc2` projections reached only 93.33% and 12.75%, while the full-width model reached 99.82%; this indicates that altered initialization and optimizer geometry, rather than lost function capacity, caused the failures.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model - 1, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc2(F.gelu(self.fc1(x[..., :-1])))
        y = y + F.pad(self.fc2_bias, (0, 1))
        return self.drop(y)
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model - 1, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.fc2._residual_gauge = True

        # Preserve the RNG stream of the original full-width Linear constructor.
        bound = 1.0 / math.sqrt(d_ff)
        torch.empty(d_ff).uniform_(-bound, bound)

        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.pad(self.fc2(F.gelu(self.fc1(x[..., :-1]))), (0, 1))
        y = y + F.pad(self.fc2_bias, (0, 1))
        return self.drop(y)
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
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if getattr(module, "_residual_gauge", False):
                full_weight = module.weight.new_empty(
                    module.out_features + 1, module.in_features
                )
                nn.init.normal_(full_weight, mean=0.0, std=0.02)
                with torch.no_grad():
                    module.weight.copy_(full_weight[:-1] - full_weight[-1:])
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
def save_json(path: Path, obj: Dict) -> None:
=======
class QuotientAdamW:
    """AdamW on seven stored differences with a virtual eighth output row."""

    def __init__(self, model: TinyDecoderLM, lr: float, weight_decay: float):
        self.all_params = list(model.parameters())
        self.gauge_params = [block.mlp.fc2.weight for block in model.blocks]
        gauge_ids = {id(param) for param in self.gauge_params}
        ordinary_params = [
            param for param in self.all_params if id(param) not in gauge_ids
        ]

        self.base = torch.optim.AdamW(
            ordinary_params, lr=lr, weight_decay=weight_decay
        )
        self.param_groups = self.base.param_groups
        self.gauge_states = [
            {
                "step": 0,
                "exp_avg": param.new_zeros(
                    param.shape[0] + 1, param.shape[1]
                ),
                "exp_avg_sq": param.new_zeros(
                    param.shape[0] + 1, param.shape[1]
                ),
            }
            for param in self.gauge_params
        ]

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base.zero_grad(set_to_none=set_to_none)
        for param in self.gauge_params:
            if set_to_none:
                param.grad = None
            elif param.grad is not None:
                param.grad.zero_()

    @torch.no_grad()
    def clip_grad_norm(self, max_norm: float) -> torch.Tensor:
        device = self.gauge_params[0].device
        total_sq = torch.zeros((), device=device)

        for param in self.all_params:
            if param.grad is not None:
                grad = param.grad.detach().float()
                total_sq.add_(grad.square().sum())

        # Include the omitted row's gradient in the norm used by the full model.
        for param in self.gauge_params:
            if param.grad is not None:
                omitted_grad = -param.grad.detach().sum(dim=0)
                total_sq.add_(omitted_grad.float().square().sum())

        total_norm = total_sq.sqrt()
        clip_coef = torch.clamp(max_norm / (total_norm + 1e-6), max=1.0)
        for param in self.all_params:
            if param.grad is not None:
                param.grad.mul_(
                    clip_coef.to(device=param.grad.device, dtype=param.grad.dtype)
                )
        return total_norm

    @torch.no_grad()
    def step(self) -> None:
        self.base.step()

        group = self.param_groups[0]
        lr = group["lr"]
        weight_decay = group["weight_decay"]
        beta1, beta2 = group["betas"]
        eps = group["eps"]

        for param, state in zip(self.gauge_params, self.gauge_states):
            if param.grad is None:
                continue

            grad = param.grad
            full_grad = torch.cat(
                [grad, -grad.sum(dim=0, keepdim=True)], dim=0
            )
            state["step"] += 1
            step = state["step"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]

            param.mul_(1.0 - lr * weight_decay)
            exp_avg.mul_(beta1).add_(full_grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(
                full_grad, full_grad, value=1.0 - beta2
            )

            bias_correction1 = 1.0 - beta1 ** step
            bias_correction2 = 1.0 - beta2 ** step
            denom = exp_avg_sq.sqrt().div_(
                math.sqrt(bias_correction2)
            ).add_(eps)
            update = exp_avg / denom
            quotient_update = update[:-1] - update[-1:]
            param.add_(
                quotient_update, alpha=-lr / bias_correction1
            )


def save_json(path: Path, obj: Dict) -> None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
=======
    optimizer = QuotientAdamW(
        model, lr=train_cfg.lr, weight_decay=train_cfg.weight_decay
    )
>>>>>>> REPLACE

<<<<<<< SEARCH
        if train_cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
=======
        if train_cfg.grad_clip > 0:
            optimizer.clip_grad_norm(train_cfg.grad_clip)
>>>>>>> REPLACE
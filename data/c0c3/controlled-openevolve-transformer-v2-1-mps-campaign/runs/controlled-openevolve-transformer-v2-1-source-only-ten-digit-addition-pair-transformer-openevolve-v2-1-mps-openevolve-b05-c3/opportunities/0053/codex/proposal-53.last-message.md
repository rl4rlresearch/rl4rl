MECHANISM: Third final-MLP output-column shift quotient

HYPOTHESIS: Gauge-fixing a third adjacent `fc2` output coefficient will reduce the model to 1,473 learned parameters while retaining at least 99% accuracy, because it contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.

INTENDED_EDIT: Extend the verified two-column `fc2` gauge to the last three hidden columns while preserving full-shape initialization, AdamW moments, weight decay, and gradient clipping.

EVIDENCE: The two-column `fc2` quotient achieved 99.93% accuracy at 1,474 parameters after the one-column version achieved 99.94%; this extends the same successful exact symmetry by one coordinate.

<<<<<<< SEARCH
class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with two common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 2
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 2))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(2),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_ff)
=======
class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with three common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 3
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 3))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(3),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedMLPProjectionLinear):
            full = module.weight.new_empty(
                module.d_model, module.d_ff
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1, -2:].clone()
                full[:, -2:].sub_(omitted)
                full[-1, -2:].zero_()
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
=======
        elif isinstance(module, GaugeFixedMLPProjectionLinear):
            full = module.weight.new_empty(
                module.d_model, module.d_ff
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1, -3:].clone()
                full[:, -3:].sub_(omitted)
                full[-1, -3:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 3 :],
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    missing_start = d_model * d_ff - 2
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(2),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -2:] = -full_grad[:-1, -2:].sum(dim=0)
    return full_grad


class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for two MLP common-output shift quotients."""
=======
def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    missing_start = d_model * d_ff - 3
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(3),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -3:] = -full_grad[:-1, -3:].sum(dim=0)
    return full_grad


class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for three MLP common-output shift quotients."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = full_value[-1, -2:].clone()
                full_value[:, -2:].sub_(omitted)
                full_value[-1, -2:].zero_()
                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 2 :],
                        )
                    )
                )
=======
                omitted = full_value[-1, -3:].clone()
                full_value[:, -3:].sub_(omitted)
                full_value[-1, -3:].zero_()
                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 3 :],
                        )
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                total_sq.add_(
                    full_grad[-1, -2:].float().square().sum()
                )
=======
                total_sq.add_(
                    full_grad[-1, -3:].float().square().sum()
                )
>>>>>>> REPLACE
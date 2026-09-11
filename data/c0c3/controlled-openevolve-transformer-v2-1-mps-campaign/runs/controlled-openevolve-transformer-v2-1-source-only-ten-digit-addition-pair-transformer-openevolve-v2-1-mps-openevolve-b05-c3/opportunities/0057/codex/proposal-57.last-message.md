MECHANISM: Seventh final-MLP common-output column gauge

HYPOTHESIS: Extending the verified six-column `fc2` gauge to a seventh adjacent hidden column will produce 1,469 learned parameters and retain at least 99% accuracy, because the omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.

INTENDED_EDIT: Gauge-fix the final output-row coefficients of the last seven `fc2` input columns while preserving full-space initialization, AdamW moments, weight decay, and gauge-aware gradient clipping.

EVIDENCE: The six-column quotient achieved 99.97% accuracy at 1,470 parameters after every one-through-five-column variant also exceeded 99%; this is another one-coordinate extension of the same repeatedly verified symmetry.

<<<<<<< SEARCH
class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with six common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 6
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 6))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(6),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_ff)
=======
class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with seven common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 7
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 7))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(7),
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
                omitted = full[-1, -6:].clone()
                full[:, -6:].sub_(omitted)
                full[-1, -6:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 6 :],
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
                omitted = full[-1, -7:].clone()
                full[:, -7:].sub_(omitted)
                full[-1, -7:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 7 :],
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
    missing_start = d_model * d_ff - 6
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(6),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -6:] = -full_grad[:-1, -6:].sum(dim=0)
    return full_grad
=======
def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    missing_start = d_model * d_ff - 7
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(7),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -7:] = -full_grad[:-1, -7:].sum(dim=0)
    return full_grad
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for six MLP common-output shift quotients."""
=======
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for seven MLP common-output shift quotients."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = full_value[-1, -6:].clone()
                full_value[:, -6:].sub_(omitted)
                full_value[-1, -6:].zero_()

                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 6 :],
                        )
                    )
                )
=======
                omitted = full_value[-1, -7:].clone()
                full_value[:, -7:].sub_(omitted)
                full_value[-1, -7:].zero_()

                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 7 :],
                        )
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                total_sq.add_(
                    full_grad[-1, -6:].float().square().sum()
                )
=======
                total_sq.add_(
                    full_grad[-1, -7:].float().square().sum()
                )
>>>>>>> REPLACE
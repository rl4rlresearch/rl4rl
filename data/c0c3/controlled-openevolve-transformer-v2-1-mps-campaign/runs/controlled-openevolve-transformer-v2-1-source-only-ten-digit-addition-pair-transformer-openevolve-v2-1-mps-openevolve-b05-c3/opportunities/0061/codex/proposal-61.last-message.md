MECHANISM: Tenth final-MLP common-output column gauge

HYPOTHESIS: Extending the verified nine-column `fc2` gauge to a tenth adjacent hidden column will reduce the model to 1,337 learned parameters while retaining at least 99% accuracy, because the omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.

INTENDED_EDIT: Gauge-fix the final output-row coefficient of the tenth trailing `fc2` input column while preserving full-space initialization, AdamW moments, weight decay, and gauge-aware gradient clipping.

EVIDENCE: The current content-independent attention model achieved 99.85% accuracy with nine `fc2` column quotients at 1,338 parameters, and every preceding one-through-nine-column extension exceeded 99%; this tests the next coordinate of the same exact symmetry.

<<<<<<< SEARCH
class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with nine common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 9
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 9))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(9),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_ff)
=======
class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with ten common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 10
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 10))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(10),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = full[-1, -9:].clone()
                full[:, -9:].sub_(omitted)
                full[-1, -9:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 9 :],
                        )
                    )
                )
=======
                omitted = full[-1, -10:].clone()
                full[:, -10:].sub_(omitted)
                full[-1, -10:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 10 :],
                        )
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    missing_start = d_model * d_ff - 9
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(9),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -9:] = -full_grad[:-1, -9:].sum(dim=0)
    return full_grad
=======
def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    missing_start = d_model * d_ff - 10
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(10),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -10:] = -full_grad[:-1, -10:].sum(dim=0)
    return full_grad
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for nine MLP common-output shift quotients."""
=======
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for ten MLP common-output shift quotients."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = full_value[-1, -9:].clone()
                full_value[:, -9:].sub_(omitted)
                full_value[-1, -9:].zero_()
                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 9 :],
                        )
                    )
                )
=======
                omitted = full_value[-1, -10:].clone()
                full_value[:, -10:].sub_(omitted)
                full_value[-1, -10:].zero_()
                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 10 :],
                        )
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                total_sq.add_(
                    full_grad[-1, -9:].float().square().sum()
                )
=======
                total_sq.add_(
                    full_grad[-1, -10:].float().square().sum()
                )
>>>>>>> REPLACE
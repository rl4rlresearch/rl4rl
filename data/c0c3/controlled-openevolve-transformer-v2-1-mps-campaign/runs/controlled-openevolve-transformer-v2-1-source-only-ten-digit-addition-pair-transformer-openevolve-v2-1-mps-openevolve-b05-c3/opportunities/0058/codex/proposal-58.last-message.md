MECHANISM: Eighth final-MLP common-output column gauge

HYPOTHESIS: Extending the verified seven-column `fc2` gauge to an eighth adjacent hidden column will produce 1,468 learned parameters and retain at least 99% accuracy because the omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.

INTENDED_EDIT: Adopt the qualified seven-column construction and gauge-fix the final output-row coefficients of the last eight `fc2` input columns while preserving full-space initialization, AdamW moments, weight decay, and gauge-aware gradient clipping.

EVIDENCE: The seven-column quotient achieved 99.95% accuracy at 1,469 parameters after every one-through-six-column variant also exceeded 99%; this is the smallest and most directly supported extension of the repeatedly verified symmetry.

<<<<<<< SEARCH
class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with five common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 5
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 5))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(5),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_ff)
=======
class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with eight common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 8
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 8))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(8),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = full[-1, -5:].clone()
                full[:, -5:].sub_(omitted)
                full[-1, -5:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 5 :],
                        )
                    )
                )
=======
                omitted = full[-1, -8:].clone()
                full[:, -8:].sub_(omitted)
                full[-1, -8:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 8 :],
                        )
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
    missing_start = d_model * d_ff - 5
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(5),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -5:] = -full_grad[:-1, -5:].sum(dim=0)
=======
    missing_start = d_model * d_ff - 8
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(8),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -8:] = -full_grad[:-1, -8:].sum(dim=0)
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for five MLP common-output shift quotients."""
=======
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for eight MLP common-output shift quotients."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = full_value[-1, -5:].clone()
                full_value[:, -5:].sub_(omitted)
                full_value[-1, -5:].zero_()

                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 5 :],
                        )
                    )
                )
=======
                omitted = full_value[-1, -8:].clone()
                full_value[:, -8:].sub_(omitted)
                full_value[-1, -8:].zero_()

                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 8 :],
                        )
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                total_sq.add_(
                    full_grad[-1, -5:].float().square().sum()
                )
=======
                total_sq.add_(
                    full_grad[-1, -8:].float().square().sum()
                )
>>>>>>> REPLACE
MECHANISM: Ninth final-MLP common-output column gauge

HYPOTHESIS: Extending the verified eight-column `fc2` gauge to a ninth adjacent hidden column will produce 1,467 learned parameters while retaining at least 99% accuracy, because the omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.

INTENDED_EDIT: Adopt the qualified eight-column construction and gauge-fix the final output-row coefficients of the last nine `fc2` input columns while preserving full-space initialization, AdamW moments, weight decay, and gauge-aware gradient clipping.

EVIDENCE: The eight-column quotient achieved 99.92% accuracy at 1,468 parameters after every one-through-seven-column variant also exceeded 99%; this is the next one-coordinate extension of the same repeatedly verified exact symmetry.

<<<<<<< SEARCH
class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with four common-output shifts fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_start = d_model * d_ff - 4
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 4))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_start],
                self.weight.new_zeros(4),
                self.weight[self.missing_start :],
            )
        )
        return flat.view(self.d_model, self.d_ff)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedMLPProjectionLinear):
            full = module.weight.new_empty(
                module.d_model, module.d_ff
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1, -4:].clone()
                full[:, -4:].sub_(omitted)
                full[-1, -4:].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_start],
                            flat[module.missing_start + 4 :],
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
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    missing_start = d_model * d_ff - 4
    full_grad = torch.cat(
        (
            parameter.grad[:missing_start],
            parameter.grad.new_zeros(4),
            parameter.grad[missing_start:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -4:] = -full_grad[:-1, -4:].sum(dim=0)
    return full_grad
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for four MLP common-output shift quotients."""
=======
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for nine MLP common-output shift quotients."""
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = full_value[-1, -4:].clone()
                full_value[:, -4:].sub_(omitted)
                full_value[-1, -4:].zero_()

                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_start],
                            flat[missing_start + 4 :],
                        )
                    )
                )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                total_sq.add_(
                    full_grad[-1, -4:].float().square().sum()
                )
=======
                total_sq.add_(
                    full_grad[-1, -9:].float().square().sum()
                )
>>>>>>> REPLACE
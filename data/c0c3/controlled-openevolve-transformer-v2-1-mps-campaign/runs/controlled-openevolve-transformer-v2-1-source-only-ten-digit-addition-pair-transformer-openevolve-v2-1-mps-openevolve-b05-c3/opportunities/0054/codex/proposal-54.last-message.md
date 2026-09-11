MECHANISM: Fourth final-MLP output-column shift quotient

HYPOTHESIS: Extending the verified three-column `fc2` gauge to a fourth adjacent hidden column will produce 1,472 learned parameters while retaining at least 99% accuracy, because the omitted coefficient contributes only a hidden-state-dependent channel-common shift removed by the final LayerNorm.

INTENDED_EDIT: Omit the final output-row coefficients of the last four `fc2` input columns while preserving full-shape initialization, virtual AdamW moments, weight decay, and gauge-aware gradient clipping.

EVIDENCE: The three-column `fc2` quotient achieved 99.98% accuracy at 1,473 parameters after the one- and two-column versions achieved 99.94% and 99.93%; this applies the same repeatedly successful exact symmetry to one additional adjacent column.

<<<<<<< SEARCH
class GaugeFixedMLPProjectionLinear(nn.Linear):
    """MLP output projection with one common-output shift fixed."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__(d_ff, d_model)
        self.d_model = d_model
        self.d_ff = d_ff
        self.missing_index = d_model * d_ff - 1
        self.weight = nn.Parameter(torch.empty(d_model * d_ff - 1))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_index],
                self.weight.new_zeros(1),
                self.weight[self.missing_index :],
            )
        )
        return flat.view(self.d_model, self.d_ff)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedMLPProjectionLinear):
            full = module.weight.new_empty(
                module.d_model, module.d_ff
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1, -1].clone()
                full[:, -1].sub_(omitted)
                full[-1, -1].zero_()
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.missing_index],
                            flat[module.missing_index + 1 :],
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
>>>>>>> REPLACE

<<<<<<< SEARCH
def full_mlp_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
    d_ff: int,
) -> torch.Tensor:
    missing_index = d_model * d_ff - 1
    full_grad = torch.cat(
        (
            parameter.grad[:missing_index],
            parameter.grad.new_zeros(1),
            parameter.grad[missing_index:],
        )
    ).view(d_model, d_ff)
    full_grad[-1, -1] = -full_grad[:-1, -1].sum()
    return full_grad
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for one MLP common-output shift quotient."""
=======
class MLPProjectionGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for four MLP common-output shift quotients."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        d_model = self.module.d_model
        d_ff = self.module.d_ff
        missing_index = self.module.missing_index
=======
        d_model = self.module.d_model
        d_ff = self.module.d_ff
        missing_start = self.module.missing_start
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = full_value[-1, -1].clone()
                full_value[:, -1].sub_(omitted)
                full_value[-1, -1].zero_()

                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:missing_index],
                            flat[missing_index + 1 :],
                        )
                    )
                )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                total_sq.add_(
                    full_grad[-1, -1].float().square()
                )
=======
                total_sq.add_(
                    full_grad[-1, -4:].float().square().sum()
                )
>>>>>>> REPLACE
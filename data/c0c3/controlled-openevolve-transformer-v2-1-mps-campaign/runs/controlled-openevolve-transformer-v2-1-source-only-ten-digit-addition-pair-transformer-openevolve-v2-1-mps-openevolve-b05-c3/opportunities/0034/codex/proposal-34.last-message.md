MECHANISM: Second LayerNorm-hyperplane key-weight quotient

HYPOTHESIS: Gauge-fixing the final input coefficient of the second key-projection row will reduce the verified model from 1599 to 1598 learned parameters while retaining at least 99% accuracy, because each key row independently has the same exact input-shift symmetry already validated for the first row.

INTENDED_EDIT: Extend the key projection representation, initialization, full-space AdamW update, and gradient clipping from one omitted key-row coefficient to two.

EVIDENCE: The current first-key-row quotient achieved 99.88% at 1599 parameters, whereas multiple direct 1599-parameter capacity ablations failed; this directly motivates applying the same successful exact symmetry to one additional key row.

<<<<<<< SEARCH
class GaugeFixedKeyLinear(nn.Linear):
    """QKV projection with one key-row coefficient fixed by LayerNorm gauge."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.missing_index = d_model * d_model + d_model - 1
        self.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 1))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight[: self.missing_index],
                self.weight.new_zeros(1),
                self.weight[self.missing_index :],
            )
        )
        return flat.view(3 * self.d_model, self.d_model)
=======
class GaugeFixedKeyLinear(nn.Linear):
    """QKV projection with two key-row coefficients fixed by LayerNorm gauge."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.missing_indices = (
            d_model * d_model + d_model - 1,
            d_model * d_model + 2 * d_model - 1,
        )
        self.weight = nn.Parameter(torch.empty(3 * d_model * d_model - 2))

    def full_weight(self) -> torch.Tensor:
        first, second = self.missing_indices
        zero = self.weight.new_zeros(1)
        flat = torch.cat(
            (
                self.weight[:first],
                zero,
                self.weight[first : second - 1],
                zero,
                self.weight[second - 1 :],
            )
        )
        return flat.view(3 * self.d_model, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        # The first key row is represented modulo the LayerNorm-hyperplane
        # direction. Key/value biases are omitted, and four query biases remain.
=======
        # The first two key rows are represented modulo the LayerNorm-hyperplane
        # direction. Key/value biases are omitted, and four query biases remain.
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedKeyLinear):
            d_model = module.d_model
            full = module.weight.new_empty(3 * d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[d_model, -1].clone()
                full[d_model, :-1].sub_(omitted)
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
        elif isinstance(module, GaugeFixedKeyLinear):
            d_model = module.d_model
            full = module.weight.new_empty(3 * d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                for row in range(d_model, d_model + 2):
                    omitted = full[row, -1].clone()
                    full[row, :-1].sub_(omitted)
                first, second = module.missing_indices
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[:first],
                            flat[first + 1 : second],
                            flat[second + 1 :],
                        )
                    )
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
def full_key_gradient(
    parameter: torch.nn.Parameter,
    ln_scale: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    missing_index = d_model * d_model + d_model - 1
    full_grad = torch.cat(
        (
            parameter.grad[:missing_index],
            parameter.grad.new_zeros(1),
            parameter.grad[missing_index:],
        )
    ).view(3 * d_model, d_model)
    gamma = ln_scale.detach()
    full_grad[d_model, -1] = -gamma[-1] * (
        full_grad[d_model, :-1] / gamma[:-1]
    ).sum()
    return full_grad
=======
def full_key_gradient(
    parameter: torch.nn.Parameter,
    ln_scale: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    first = d_model * d_model + d_model - 1
    second = d_model * d_model + 2 * d_model - 1
    zero = parameter.grad.new_zeros(1)
    full_grad = torch.cat(
        (
            parameter.grad[:first],
            zero,
            parameter.grad[first : second - 1],
            zero,
            parameter.grad[second - 1 :],
        )
    ).view(3 * d_model, d_model)
    gamma = ln_scale.detach()
    for row in range(d_model, d_model + 2):
        full_grad[row, -1] = -gamma[-1] * (
            full_grad[row, :-1] / gamma[:-1]
        ).sum()
    return full_grad
>>>>>>> REPLACE

<<<<<<< SEARCH
class KeyGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for one LayerNorm-induced key-weight quotient."""
=======
class KeyGaugeAdamW(torch.optim.Optimizer):
    """Full-space AdamW for LayerNorm-induced key-weight quotients."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        d_model = self.module.d_model
        missing_index = self.module.missing_index
=======
        d_model = self.module.d_model
        first, second = self.module.missing_indices
>>>>>>> REPLACE

<<<<<<< SEARCH
                gamma = self.ln_scale.detach()
                key_row = full_value[d_model]
                omitted = key_row[-1].clone()
                key_row[:-1].sub_(
                    omitted * gamma[-1] / gamma[:-1]
                )
                key_row[-1].zero_()

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
                gamma = self.ln_scale.detach()
                for row in range(d_model, d_model + 2):
                    key_row = full_value[row]
                    omitted = key_row[-1].clone()
                    key_row[:-1].sub_(
                        omitted * gamma[-1] / gamma[:-1]
                    )
                    key_row[-1].zero_()

                flat = full_value.reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            flat[:first],
                            flat[first + 1 : second],
                            flat[second + 1 :],
                        )
                    )
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                total_sq.add_(
                    full_grad[d_model, -1].float().square()
                )
=======
                total_sq.add_(
                    full_grad[d_model : d_model + 2, -1]
                    .float()
                    .square()
                    .sum()
                )
>>>>>>> REPLACE
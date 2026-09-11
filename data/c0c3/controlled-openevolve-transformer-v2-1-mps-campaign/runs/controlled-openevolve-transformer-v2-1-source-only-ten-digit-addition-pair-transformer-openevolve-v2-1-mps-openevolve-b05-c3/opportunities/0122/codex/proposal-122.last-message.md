MECHANISM: Two-feature value/projection scale quotient

HYPOTHESIS: Extending the qualified single-feature attention scale quotient to a second independent value feature will reduce the model from 1,123 to 1,122 learned parameters while retaining at least 99% accuracy, because both removed coefficients are exact multiplicative factorization gauges and no attention feature is tied or deleted.

INTENDED_EDIT: Fix the first two attention-projection coefficients, absorb their scales into the corresponding learned value rows at initialization and after optimizer steps, and reconstruct the two omitted projection gradients.

EVIDENCE: The single-feature value/projection scale quotient achieved 99.96% accuracy with 1,123 parameters, while attention-bias tying and whole value-encoder sharing failed; this motivates extending the successful exact symmetry removal by one independent feature without reducing functional capacity.

<<<<<<< SEARCH
class GaugeFixedProjectionLinear(nn.Linear):
    """Attention projection with all common-output shifts fixed."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.missing_start = (d_model - 1) * d_model
        self.weight = nn.Parameter(torch.empty((d_model - 1) * d_model))

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.weight,
                self.weight.new_zeros(self.d_model),
            )
        )
        return flat.view(self.d_model, self.d_model)
=======
class GaugeFixedProjectionLinear(nn.Linear):
    """Attention projection modulo output shifts and two value-scale gauges."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.register_buffer("fixed_anchor", torch.full((2,), 0.02))
        self.weight = nn.Parameter(
            torch.empty((d_model - 1) * d_model - 2)
        )
        object.__setattr__(self, "_value_source", None)

    def share_value(self, source: nn.Module) -> None:
        object.__setattr__(self, "_value_source", source)

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.fixed_anchor,
                self.weight,
                self.weight.new_zeros(self.d_model),
            )
        )
        return flat.view(self.d_model, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.value = GaugeFixedValueLinear(d_model)
        self.proj = GaugeFixedProjectionLinear(d_model)
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
=======
        self.value = GaugeFixedValueLinear(d_model)
        self.proj = GaugeFixedProjectionLinear(d_model)
        self.proj.share_value(self.value)
        self.proj.bias = nn.Parameter(torch.empty(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)

    @staticmethod
=======
        self.apply(self._init_weights)
        with torch.no_grad():
            for block in self.blocks:
                block.attn.value.weight[:2].mul_(
                    block.attn.proj._value_init_scale[:, None]
                )

    @staticmethod
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedProjectionLinear):
            d_model = module.d_model
            full = module.weight.new_empty(d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1].clone()
                full.sub_(omitted)
                full[-1].zero_()
                module.weight.copy_(full[:-1].reshape(-1))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
=======
        elif isinstance(module, GaugeFixedProjectionLinear):
            d_model = module.d_model
            full = module.weight.new_empty(d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1].clone()
                full.sub_(omitted)
                full[-1].zero_()
                anchors = full[0, :2].clone()
                scale = module.fixed_anchor / anchors
                full[:, :2].mul_(scale)
                module.weight.copy_(full[:-1].reshape(-1)[2:])
                module._value_init_scale = scale.reciprocal().detach()
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
def full_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    full_grad = torch.cat(
        (
            parameter.grad,
            parameter.grad.new_zeros(d_model),
        )
    ).view(d_model, d_model)
    full_grad[-1] = -full_grad[:-1].sum(dim=0)
    return full_grad
=======
def full_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    full_grad = torch.cat(
        (
            parameter.grad.new_zeros(2),
            parameter.grad,
            parameter.grad.new_zeros(d_model),
        )
    ).view(d_model, d_model)
    full_grad[-1] = -full_grad[:-1].sum(dim=0)
    return full_grad
>>>>>>> REPLACE

<<<<<<< SEARCH
                omitted = full_value[-1].clone()
                full_value.sub_(omitted)
                full_value[-1].zero_()
                parameter.copy_(full_value[:-1].reshape(-1))

        return loss


def full_mlp_projection_gradient(
=======
                omitted = full_value[-1].clone()
                full_value.sub_(omitted)
                full_value[-1].zero_()

                anchors = full_value[0, :2].clone()
                scale = self.module.fixed_anchor / anchors
                full_value[:, :2].mul_(scale)
                self.module._value_source.weight[:2].div_(
                    scale[:, None]
                )
                parameter.copy_(full_value[:-1].reshape(-1)[2:])

        return loss


def full_mlp_projection_gradient(
>>>>>>> REPLACE
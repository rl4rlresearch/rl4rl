MECHANISM: Single-feature value/projection scale quotient

HYPOTHESIS: Fixing one attention-output projection coefficient and absorbing its scale into the corresponding learned value feature will reduce the qualified model from 1,124 to 1,123 parameters while retaining at least 99% accuracy, because this removes an exact factorization symmetry without tying or deleting any attention feature.

INTENDED_EDIT: Canonically fix the first attention-projection coefficient, reconstruct the projection from 55 learned coordinates, rescale the matching value row at initialization and after optimizer steps, and adapt gauge-gradient reconstruction.

EVIDENCE: The 1,124-parameter model reached 99.84%–99.85%, while independently tying attention or MLP output biases failed and sharing whole value encoders collapsed to 6.31%; an exact one-coordinate value/projection scale quotient preserves those sensitive functional degrees of freedom.

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
    """Attention projection modulo output shifts and one value-scale gauge."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.register_buffer("fixed_anchor", torch.tensor(0.02))
        self.weight = nn.Parameter(
            torch.empty((d_model - 1) * d_model - 1)
        )
        object.__setattr__(self, "_value_source", None)

    def share_value(self, source: nn.Module) -> None:
        object.__setattr__(self, "_value_source", source)

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.fixed_anchor.reshape(1),
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
                block.attn.value.weight[0].mul_(
                    block.attn.proj._value_init_scale
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
                anchor = full[0, 0].clone()
                scale = module.fixed_anchor / anchor
                full[:, 0].mul_(scale)
                module.weight.copy_(full[:-1].reshape(-1)[1:])
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
            parameter.grad.new_zeros(1),
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

                anchor = full_value[0, 0].clone()
                scale = self.module.fixed_anchor / anchor
                full_value[:, 0].mul_(scale)
                self.module._value_source.weight[0].div_(scale)
                parameter.copy_(full_value[:-1].reshape(-1)[1:])

        return loss


def full_mlp_projection_gradient(
>>>>>>> REPLACE
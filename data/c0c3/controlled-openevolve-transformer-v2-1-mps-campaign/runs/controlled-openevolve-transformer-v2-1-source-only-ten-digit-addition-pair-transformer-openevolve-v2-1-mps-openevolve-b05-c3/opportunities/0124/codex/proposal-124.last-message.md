MECHANISM: Conditioned three-feature value/projection scale quotient

HYPOTHESIS: Directly initializing the third fixed projection anchor in its canonical chart will reduce the qualified 1,122-parameter design to 1,121 parameters while retaining at least 99% accuracy, avoiding the potentially ill-conditioned initialization rescaling used by the failed three-feature attempt.

INTENDED_EDIT: Fully fix the final LayerNorm bias, remove three attention-projection coefficients, absorb their scales into the corresponding learned value features during optimization, preserve the qualified initialization for the first two features, and directly initialize the new third anchor.

EVIDENCE: The two-feature quotient achieved 99.98% accuracy with 1,122 parameters, whereas the three-feature extension collapsed to 0.0001 accuracy; changing only the new feature’s chart initialization tests whether that failure was optimization conditioning rather than loss of functional capacity.

<<<<<<< SEARCH
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with shared scale and five-way-tied zero-sum bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(d_model - 5))

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)

    def full_bias(self) -> torch.Tensor:
        anchored = torch.cat((self.bias, self.bias.new_zeros(5)))
        return anchored - anchored.mean()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            self.full_bias(),
            self.eps,
        )
=======
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with shared scale and fully tied zero-sum bias."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(1))
        self.bias = None

    def full_weight(self) -> torch.Tensor:
        return self.weight.expand(self.normalized_shape)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            None,
            self.eps,
        )
>>>>>>> REPLACE

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
    """Attention projection modulo output shifts and three value-scale gauges."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.register_buffer("fixed_anchor", torch.full((3,), 0.02))
        self.weight = nn.Parameter(
            torch.empty((d_model - 1) * d_model - 3)
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
    def _init_weights(module: nn.Module) -> None:
=======
        self.apply(self._init_weights)
        with torch.no_grad():
            for block in self.blocks:
                block.attn.value.weight[:2].mul_(
                    block.attn.proj._value_init_scale[:, None]
                )

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
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
                scale = module.fixed_anchor[:2] / anchors
                full[:, :2].mul_(scale)
                full[0, 2] = module.fixed_anchor[2]
                module.weight.copy_(full[:-1].reshape(-1)[3:])
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
            parameter.grad.new_zeros(3),
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

                anchors = full_value[0, :3].clone()
                scale = self.module.fixed_anchor / anchors
                full_value[:, :3].mul_(scale)
                self.module._value_source.weight[:3].div_(
                    scale[:, None]
                )
                parameter.copy_(full_value[:-1].reshape(-1)[3:])

        return loss


def full_mlp_projection_gradient(
>>>>>>> REPLACE

<<<<<<< SEARCH
    relative_bias_gauge_parameters = [
        block.attn.relative_bias for block in model.blocks
    ]
    final_bias_gauge_parameters = [
        model.ln_f.bias
    ]
    last_coordinate_gauge_parameters = [
        *embedding_gauge_parameters,
        *output_bias_gauge_parameters,
        *attention_output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
        *final_bias_gauge_parameters,
    ]
=======
    relative_bias_gauge_parameters = [
        block.attn.relative_bias for block in model.blocks
    ]
    last_coordinate_gauge_parameters = [
        *embedding_gauge_parameters,
        *output_bias_gauge_parameters,
        *attention_output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
    ]
>>>>>>> REPLACE
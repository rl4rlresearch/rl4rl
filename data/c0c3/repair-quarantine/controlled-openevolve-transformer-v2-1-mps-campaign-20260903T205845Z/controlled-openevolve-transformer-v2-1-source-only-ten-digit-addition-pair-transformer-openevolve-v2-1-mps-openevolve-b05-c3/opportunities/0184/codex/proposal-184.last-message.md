MECHANISM: Extended compensated within-head projection gauge fixing

HYPOTHESIS: Removing the final LayerNorm bias and fixing seven projection coordinates—including one additional first-head shear—will reduce the model from 1,127 to 1,117 parameters while retaining accuracy >= 0.99 after 4,999 training steps.

INTENDED_EDIT: Apply the qualified bias-free, six-coordinate projection parameterization, then eliminate one additional first-head projection shear with matching value-basis compensation during initialization and every optimizer step.

EVIDENCE: All three 1,118-parameter reference designs achieved 0.9998–0.9999 accuracy using a bias-free final LayerNorm and compensated projection gauges; the extra removed coordinate is another exact within-head projection/value basis shear.

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
    """Projection modulo output shifts, three scales, and four shears."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.second_missing = d_model // 2 + 1
        self.register_buffer(
            "fixed_anchor",
            torch.tensor([0.02, 0.02, 0.0, 0.0, 0.02]),
        )
        self.weight = nn.Parameter(
            torch.empty((d_model - 1) * d_model - 7)
        )
        object.__setattr__(self, "_value_source", None)

    def share_value(self, source: nn.Module) -> None:
        object.__setattr__(self, "_value_source", source)

    def full_weight(self) -> torch.Tensor:
        prefix_count = self.second_missing - 5
        flat = torch.cat(
            (
                self.fixed_anchor[:4],
                self.weight[:prefix_count],
                self.fixed_anchor[4:],
                self.weight.new_zeros(2),
                self.weight[prefix_count:],
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
                projection = block.attn.proj
                value = block.attn.value
                value.weight[:2].mul_(
                    projection._value_init_scale[:, None]
                )
                value.weight[0].add_(
                    (
                        projection._value_init_shear[:, None]
                        * value.weight[2:4]
                    ).sum(dim=0)
                )

                head_start = projection.second_missing - 1
                head_value = value.weight[head_start:].clone()
                value.weight[head_start:].copy_(
                    head_value[
                        projection._value_init_head_permutation
                    ]
                )
                value.weight[head_start].mul_(
                    projection._value_init_head_scale
                )
                value.weight[head_start].add_(
                    (
                        projection._value_init_head_shear[:, None]
                        * value.weight[
                            projection.second_missing :
                            projection.second_missing + 2
                        ]
                    ).sum(dim=0)
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
        elif isinstance(module, GaugeFixedMLPProjectionLinear):
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
                shear = full[0, 2:4] / full[0, 0]
                full[:, 2:4].sub_(
                    full[:, :1] * shear
                )

                head_start = module.second_missing - 1
                head_order = torch.argsort(
                    full[0, head_start:].abs(),
                    descending=True,
                )
                head_permutation = torch.cat(
                    (
                        head_order[:1],
                        head_order[-1:],
                        head_order[1:-1],
                    )
                )
                permuted_head = full[:, head_start:].clone()[
                    :, head_permutation
                ]
                full[:, head_start:].copy_(permuted_head)
                head_anchor = full[0, head_start].clone()
                head_scale = module.fixed_anchor[4] / head_anchor
                full[:, head_start].mul_(head_scale)
                head_shear = (
                    full[
                        0,
                        module.second_missing :
                        module.second_missing + 2,
                    ]
                    / full[0, head_start]
                )
                full[
                    :,
                    module.second_missing :
                    module.second_missing + 2,
                ].sub_(
                    full[:, head_start : head_start + 1]
                    * head_shear
                )

                reduced = full[:-1].reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            reduced[4:head_start],
                            reduced[module.second_missing + 2 :],
                        )
                    )
                )
                module._value_init_scale = scale.reciprocal().detach()
                module._value_init_shear = shear.detach()
                module._value_init_head_permutation = (
                    head_permutation.detach()
                )
                module._value_init_head_scale = (
                    head_scale.reciprocal().detach()
                )
                module._value_init_head_shear = head_shear.detach()
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedMLPProjectionLinear):
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
    second_missing = d_model // 2 + 1
    prefix_count = second_missing - 5
    full_grad = torch.cat(
        (
            parameter.grad.new_zeros(4),
            parameter.grad[:prefix_count],
            parameter.grad.new_zeros(3),
            parameter.grad[prefix_count:],
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
                scale = self.module.fixed_anchor[:2] / anchors
                full_value[:, :2].mul_(scale)
                self.module._value_source.weight[:2].div_(
                    scale[:, None]
                )

                shear = full_value[0, 2:4] / full_value[0, 0]
                full_value[:, 2:4].sub_(
                    full_value[:, :1] * shear
                )
                self.module._value_source.weight[0].add_(
                    (
                        shear[:, None]
                        * self.module._value_source.weight[2:4]
                    ).sum(dim=0)
                )

                head_start = self.module.second_missing - 1
                head_anchor = full_value[0, head_start].clone()
                head_scale = self.module.fixed_anchor[4] / head_anchor
                full_value[:, head_start].mul_(head_scale)
                self.module._value_source.weight[head_start].div_(
                    head_scale
                )
                head_shear = (
                    full_value[
                        0,
                        self.module.second_missing :
                        self.module.second_missing + 2,
                    ]
                    / full_value[0, head_start]
                )
                full_value[
                    :,
                    self.module.second_missing :
                    self.module.second_missing + 2,
                ].sub_(
                    full_value[:, head_start : head_start + 1]
                    * head_shear
                )
                self.module._value_source.weight[head_start].add_(
                    (
                        head_shear[:, None]
                        * self.module._value_source.weight[
                            self.module.second_missing :
                            self.module.second_missing + 2
                        ]
                    ).sum(dim=0)
                )

                reduced = full_value[:-1].reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            reduced[4:head_start],
                            reduced[self.module.second_missing + 2 :],
                        )
                    )
                )

        return loss


def full_mlp_projection_gradient(
>>>>>>> REPLACE

<<<<<<< SEARCH
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
    last_coordinate_gauge_parameters = [
        *embedding_gauge_parameters,
        *output_bias_gauge_parameters,
        *attention_output_bias_gauge_parameters,
        *relative_bias_gauge_parameters,
    ]
>>>>>>> REPLACE
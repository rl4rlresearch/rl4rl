MECHANISM: Six-terminal kernel tie atop nested orthogonal attention quotients

HYPOTHESIS: Extending the verified 1,116-parameter nested-rotation/five-terminal-kernel design to six tied terminal logits will reduce the model to 1,115 parameters while retaining at least 99% accuracy, because it preserves the qualified projection geometry and removes only one additional relative-kernel degree of freedom.

INTENDED_EDIT: Adopt the verified nested scale/shear/Givens projection chart and fully tied final LayerNorm bias, then shorten the learned relative-bias vector by one and reconstruct six tied terminal logits.

EVIDENCE: The nested orthogonal quotient achieved 99.99% at 1,117 parameters, and adding the fifth terminal kernel tie improved the resulting 1,116-parameter design to 100%; projection-side attempts at 1,116 collapsed, so extending the successful kernel tie is the most informative next reduction.

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
    """Projection modulo shifts, scales, shears, and nested rotations."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.head_start = d_model // 2
        self.second_missing = self.head_start + 1
        self.second_rotation_missing = self.head_start + 2
        self.nested_missing = d_model + self.second_missing
        self.register_buffer(
            "fixed_anchor", torch.tensor([0.02, 0.02, 0.0])
        )
        self.register_buffer(
            "fixed_head_anchor", torch.tensor(0.02)
        )
        self.weight = nn.Parameter(
            torch.empty((d_model - 1) * d_model - 7)
        )
        object.__setattr__(self, "_value_source", None)

    def share_value(self, source: nn.Module) -> None:
        object.__setattr__(self, "_value_source", source)

    def full_weight(self) -> torch.Tensor:
        prefix_count = self.head_start - 3
        nested_prefix_count = self.nested_missing - 6
        flat = torch.cat(
            (
                self.fixed_anchor,
                self.weight[:prefix_count],
                self.fixed_head_anchor.reshape(1),
                self.weight.new_zeros(2),
                self.weight[
                    prefix_count:nested_prefix_count
                ],
                self.weight.new_zeros(1),
                self.weight[nested_prefix_count:],
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
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the four terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 4)
        )
=======
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the six terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 6)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias,
                self.relative_bias.new_zeros(4),
=======
                self.relative_bias,
                self.relative_bias.new_zeros(6),
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
                    projection._value_init_shear * value.weight[2]
                )

                head_start = projection.head_start
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
                    projection._value_init_head_shear
                    * value.weight[head_start + 1]
                )

                rotation_start = projection.second_rotation_missing
                rotated_value = value.weight[
                    rotation_start : rotation_start + 2
                ].clone()
                value.weight[
                    rotation_start : rotation_start + 2
                ].copy_(
                    projection._value_init_head_rotation
                    @ rotated_value
                )

                nested_start = projection.second_missing
                nested_value = value.weight[
                    nested_start : nested_start + 2
                ].clone()
                value.weight[
                    nested_start : nested_start + 2
                ].copy_(
                    projection._value_init_nested_rotation
                    @ nested_value
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
                scale = module.fixed_anchor[:2] / anchors
                full[:, :2].mul_(scale)
                shear = full[0, 2] / full[0, 0]
                full[:, 2].sub_(shear * full[:, 0])

                head_start = module.head_start
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
                head_scale = module.fixed_head_anchor / head_anchor
                full[:, head_start].mul_(head_scale)
                head_shear = (
                    full[0, module.second_missing]
                    / full[0, head_start]
                )
                full[:, module.second_missing].sub_(
                    head_shear * full[:, head_start]
                )

                rotation_start = module.second_rotation_missing
                rotation_pair = full[
                    0, rotation_start : rotation_start + 2
                ]
                rotation_norm = rotation_pair.norm()
                head_rotation = torch.stack(
                    (
                        rotation_pair[1] / rotation_norm,
                        -rotation_pair[0] / rotation_norm,
                        rotation_pair[0] / rotation_norm,
                        rotation_pair[1] / rotation_norm,
                    )
                ).view(2, 2)
                full[
                    :, rotation_start : rotation_start + 2
                ].copy_(
                    full[
                        :, rotation_start : rotation_start + 2
                    ].clone()
                    @ head_rotation.transpose(0, 1)
                )

                nested_start = module.second_missing
                nested_pair = full[
                    1, nested_start : nested_start + 2
                ]
                nested_norm = nested_pair.norm()
                nested_rotation = torch.stack(
                    (
                        nested_pair[1] / nested_norm,
                        -nested_pair[0] / nested_norm,
                        nested_pair[0] / nested_norm,
                        nested_pair[1] / nested_norm,
                    )
                ).view(2, 2)
                full[
                    :, nested_start : nested_start + 2
                ].copy_(
                    full[
                        :, nested_start : nested_start + 2
                    ].clone()
                    @ nested_rotation.transpose(0, 1)
                )

                reduced = full[:-1].reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            reduced[3 : module.head_start],
                            reduced[
                                module.second_rotation_missing + 1 :
                                module.nested_missing
                            ],
                            reduced[module.nested_missing + 1 :],
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
                module._value_init_head_rotation = (
                    head_rotation.detach()
                )
                module._value_init_nested_rotation = (
                    nested_rotation.detach()
                )
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
    head_start = d_model // 2
    prefix_count = head_start - 3
    nested_missing = d_model + head_start + 1
    nested_prefix_count = nested_missing - 6
    full_grad = torch.cat(
        (
            parameter.grad.new_zeros(3),
            parameter.grad[:prefix_count],
            parameter.grad.new_zeros(3),
            parameter.grad[
                prefix_count:nested_prefix_count
            ],
            parameter.grad.new_zeros(1),
            parameter.grad[nested_prefix_count:],
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

                shear = full_value[0, 2] / full_value[0, 0]
                full_value[:, 2].sub_(
                    shear * full_value[:, 0]
                )
                self.module._value_source.weight[0].add_(
                    shear * self.module._value_source.weight[2]
                )

                head_start = self.module.head_start
                head_scale = (
                    self.module.fixed_head_anchor
                    / full_value[0, head_start]
                )
                full_value[:, head_start].mul_(head_scale)
                self.module._value_source.weight[head_start].div_(
                    head_scale
                )
                head_shear = (
                    full_value[0, self.module.second_missing]
                    / full_value[0, head_start]
                )
                full_value[:, self.module.second_missing].sub_(
                    head_shear * full_value[:, head_start]
                )
                self.module._value_source.weight[head_start].add_(
                    head_shear
                    * self.module._value_source.weight[
                        self.module.second_missing
                    ]
                )

                rotation_start = self.module.second_rotation_missing
                rotation_pair = full_value[
                    0, rotation_start : rotation_start + 2
                ]
                rotation_norm = rotation_pair.norm()
                head_rotation = torch.stack(
                    (
                        rotation_pair[1] / rotation_norm,
                        -rotation_pair[0] / rotation_norm,
                        rotation_pair[0] / rotation_norm,
                        rotation_pair[1] / rotation_norm,
                    )
                ).view(2, 2)
                full_value[
                    :, rotation_start : rotation_start + 2
                ].copy_(
                    full_value[
                        :, rotation_start : rotation_start + 2
                    ].clone()
                    @ head_rotation.transpose(0, 1)
                )
                rotated_value = self.module._value_source.weight[
                    rotation_start : rotation_start + 2
                ].clone()
                self.module._value_source.weight[
                    rotation_start : rotation_start + 2
                ].copy_(head_rotation @ rotated_value)

                nested_start = self.module.second_missing
                nested_pair = full_value[
                    1, nested_start : nested_start + 2
                ]
                nested_norm = nested_pair.norm()
                nested_rotation = torch.stack(
                    (
                        nested_pair[1] / nested_norm,
                        -nested_pair[0] / nested_norm,
                        nested_pair[0] / nested_norm,
                        nested_pair[1] / nested_norm,
                    )
                ).view(2, 2)
                full_value[
                    :, nested_start : nested_start + 2
                ].copy_(
                    full_value[
                        :, nested_start : nested_start + 2
                    ].clone()
                    @ nested_rotation.transpose(0, 1)
                )
                nested_value = self.module._value_source.weight[
                    nested_start : nested_start + 2
                ].clone()
                self.module._value_source.weight[
                    nested_start : nested_start + 2
                ].copy_(nested_rotation @ nested_value)

                reduced = full_value[:-1].reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            reduced[3 : self.module.head_start],
                            reduced[
                                self.module.second_rotation_missing + 1 :
                                self.module.nested_missing
                            ],
                            reduced[self.module.nested_missing + 1 :],
                        )
                    )
                )

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
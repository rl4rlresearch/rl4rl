MECHANISM: Additional second-head value-basis shear gauge fixing

HYPOTHESIS: Eliminating a second projection coefficient in the second attention head through the same compensated shear symmetry will reduce the qualified 1,120-parameter design to 1,119 parameters while retaining accuracy >= 0.99.

INTENDED_EDIT: Extend the verified two-head projection gauge with another zeroed second-head coordinate, including exact initialization compensation, gradient reconstruction, and optimizer regauging.

EVIDENCE: Reference Design 3 achieved 1.0 accuracy with 1,120 parameters after 4,999 steps using the same largest-magnitude second-head pivot for one compensated shear; the additional shear removes another coordinate without changing the represented attention function.

<<<<<<< SEARCH
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
=======
class GaugeFixedProjectionLinear(nn.Linear):
    """Projection modulo output shifts, two scales, and three shears."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.second_missing = d_model // 2 + 1
        self.register_buffer(
            "fixed_anchor", torch.tensor([0.02, 0.02, 0.0])
        )
        self.weight = nn.Parameter(
            torch.empty((d_model - 1) * d_model - 5)
        )
        object.__setattr__(self, "_value_source", None)

    def share_value(self, source: nn.Module) -> None:
        object.__setattr__(self, "_value_source", source)

    def full_weight(self) -> torch.Tensor:
        prefix_count = self.second_missing - 3
        flat = torch.cat(
            (
                self.fixed_anchor,
                self.weight[:prefix_count],
                self.weight.new_zeros(2),
                self.weight[prefix_count:],
                self.weight.new_zeros(self.d_model),
            )
        )
        return flat.view(self.d_model, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.apply(self._init_weights)
        with torch.no_grad():
            for block in self.blocks:
                block.attn.value.weight[0].mul_(
                    block.attn.proj._value_init_scale
                )
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

                head_start = projection.second_missing - 1
                head_value = value.weight[head_start:].clone()
                value.weight[head_start:].copy_(
                    head_value[
                        projection._value_init_head_permutation
                    ]
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
                anchor = full[0, 0].clone()
                scale = module.fixed_anchor / anchor
                full[:, 0].mul_(scale)
                module.weight.copy_(full[:-1].reshape(-1)[1:])
                module._value_init_scale = scale.reciprocal().detach()
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
                            reduced[3 : module.second_missing],
                            reduced[module.second_missing + 2 :],
                        )
                    )
                )
                module._value_init_scale = scale.reciprocal().detach()
                module._value_init_shear = shear.detach()
                module._value_init_head_permutation = (
                    head_permutation.detach()
                )
                module._value_init_head_shear = head_shear.detach()
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
            parameter.grad.new_zeros(1),
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
    prefix_count = second_missing - 3
    full_grad = torch.cat(
        (
            parameter.grad.new_zeros(3),
            parameter.grad[:prefix_count],
            parameter.grad.new_zeros(2),
            parameter.grad[prefix_count:],
            parameter.grad.new_zeros(d_model),
        )
    ).view(d_model, d_model)
    full_grad[-1] = -full_grad[:-1].sum(dim=0)
    return full_grad
>>>>>>> REPLACE

<<<<<<< SEARCH
                anchor = full_value[0, 0].clone()
                scale = self.module.fixed_anchor / anchor
                full_value[:, 0].mul_(scale)
                self.module._value_source.weight[0].div_(scale)
                parameter.copy_(full_value[:-1].reshape(-1)[1:])
=======
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

                head_start = self.module.second_missing - 1
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
                            reduced[3 : self.module.second_missing],
                            reduced[self.module.second_missing + 2 :],
                        )
                    )
                )
>>>>>>> REPLACE
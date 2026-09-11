MECHANISM: Residual second-head value-basis shear gauge fixing

HYPOTHESIS: Starting from the qualified 1,118-parameter projection gauge, eliminating one row-1 projection coefficient through a compensated shear between the two second-head columns already zero in row 0 will produce a 1,117-parameter model with accuracy >= 0.99.

INTENDED_EDIT: Adopt the qualified fixed second-head pivot and two-shear parameterization, then add a magnitude-pivoted secondary shear with exact initialization compensation, gradient reconstruction, and optimizer regauging.

EVIDENCE: Reference Design 2 achieved 0.9999 accuracy with 1,118 parameters after 4,999 steps. Unlike the failed removal of the remaining first-row coefficient, this patch retains that coefficient and uses the residual basis symmetry between two columns whose first-row entries are already fixed to zero.

<<<<<<< SEARCH
class GaugeFixedProjectionLinear(nn.Linear):
    """Projection modulo output shifts, two scales, and one shear per head."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.second_missing = d_model // 2 + 1
        self.register_buffer(
            "fixed_anchor", torch.tensor([0.02, 0.02, 0.0])
        )
        self.weight = nn.Parameter(
            torch.empty((d_model - 1) * d_model - 4)
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
                self.weight.new_zeros(1),
                self.weight[prefix_count:],
                self.weight.new_zeros(self.d_model),
            )
        )
        return flat.view(self.d_model, self.d_model)
=======
class GaugeFixedProjectionLinear(nn.Linear):
    """Projection modulo output shifts and within-head basis gauges."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.second_missing = d_model // 2 + 1
        self.secondary_missing = (
            d_model + self.second_missing + 1
        )
        self.register_buffer(
            "fixed_anchor", torch.tensor([0.02, 0.02, 0.0, 0.02])
        )
        self.weight = nn.Parameter(
            torch.empty((d_model - 1) * d_model - 7)
        )
        object.__setattr__(self, "_value_source", None)

    def share_value(self, source: nn.Module) -> None:
        object.__setattr__(self, "_value_source", source)

    def full_weight(self) -> torch.Tensor:
        prefix_count = self.second_missing - 4
        secondary_prefix_count = (
            self.secondary_missing - self.second_missing - 2
        )
        secondary_end = prefix_count + secondary_prefix_count
        flat = torch.cat(
            (
                self.fixed_anchor[:3],
                self.weight[:prefix_count],
                self.fixed_anchor[3:],
                self.weight.new_zeros(2),
                self.weight[prefix_count:secondary_end],
                self.weight.new_zeros(1),
                self.weight[secondary_end:],
                self.weight.new_zeros(self.d_model),
            )
        )
        return flat.view(self.d_model, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                head_start = projection.second_missing - 1
                head_value = value.weight[head_start:].clone()
                value.weight[head_start:].copy_(
                    head_value[
                        projection._value_init_head_permutation
                    ]
                )
                value.weight[head_start].add_(
                    projection._value_init_head_shear
                    * value.weight[head_start + 1]
                )
=======
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

                secondary_start = projection.second_missing
                secondary_value = value.weight[
                    secondary_start : secondary_start + 2
                ].clone()
                value.weight[
                    secondary_start : secondary_start + 2
                ].copy_(
                    secondary_value[
                        projection._value_init_secondary_permutation
                    ]
                )
                value.weight[secondary_start].add_(
                    projection._value_init_secondary_shear
                    * value.weight[secondary_start + 1]
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
                    full[0, module.second_missing]
                    / full[0, head_start]
                )
                full[:, module.second_missing].sub_(
                    head_shear * full[:, head_start]
                )
=======
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
                head_scale = module.fixed_anchor[3] / head_anchor
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

                secondary_start = module.second_missing
                secondary_order = torch.argsort(
                    full[
                        1,
                        secondary_start : secondary_start + 2,
                    ].abs(),
                    descending=True,
                )
                secondary_head = full[
                    :, secondary_start : secondary_start + 2
                ].clone()[:, secondary_order]
                full[
                    :, secondary_start : secondary_start + 2
                ].copy_(secondary_head)
                secondary_shear = (
                    full[1, secondary_start + 1]
                    / full[1, secondary_start]
                )
                full[:, secondary_start + 1].sub_(
                    secondary_shear * full[:, secondary_start]
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                reduced = full[:-1].reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            reduced[3 : module.second_missing],
                            reduced[module.second_missing + 1 :],
                        )
                    )
                )
                module._value_init_scale = scale.reciprocal().detach()
                module._value_init_shear = shear.detach()
                module._value_init_head_permutation = (
                    head_permutation.detach()
                )
                module._value_init_head_shear = head_shear.detach()
=======
                reduced = full[:-1].reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            reduced[3:head_start],
                            reduced[
                                module.second_missing + 2 :
                                module.secondary_missing
                            ],
                            reduced[module.secondary_missing + 1 :],
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
                module._value_init_secondary_permutation = (
                    secondary_order.detach()
                )
                module._value_init_secondary_shear = (
                    secondary_shear.detach()
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
            parameter.grad.new_zeros(1),
            parameter.grad[prefix_count:],
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
    secondary_missing = d_model + second_missing + 1
    prefix_count = second_missing - 4
    secondary_prefix_count = (
        secondary_missing - second_missing - 2
    )
    secondary_end = prefix_count + secondary_prefix_count
    full_grad = torch.cat(
        (
            parameter.grad.new_zeros(3),
            parameter.grad[:prefix_count],
            parameter.grad.new_zeros(3),
            parameter.grad[prefix_count:secondary_end],
            parameter.grad.new_zeros(1),
            parameter.grad[secondary_end:],
            parameter.grad.new_zeros(d_model),
        )
    ).view(d_model, d_model)
    full_grad[-1] = -full_grad[:-1].sum(dim=0)
    return full_grad
>>>>>>> REPLACE

<<<<<<< SEARCH
                head_start = self.module.second_missing - 1
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

                reduced = full_value[:-1].reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            reduced[3 : self.module.second_missing],
                            reduced[self.module.second_missing + 1 :],
                        )
                    )
                )
=======
                head_start = self.module.second_missing - 1
                head_anchor = full_value[0, head_start].clone()
                head_scale = self.module.fixed_anchor[3] / head_anchor
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

                secondary_start = self.module.second_missing
                secondary_shear = (
                    full_value[1, secondary_start + 1]
                    / full_value[1, secondary_start]
                )
                full_value[:, secondary_start + 1].sub_(
                    secondary_shear
                    * full_value[:, secondary_start]
                )
                self.module._value_source.weight[
                    secondary_start
                ].add_(
                    secondary_shear
                    * self.module._value_source.weight[
                        secondary_start + 1
                    ]
                )

                reduced = full_value[:-1].reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            reduced[3:head_start],
                            reduced[
                                self.module.second_missing + 2 :
                                self.module.secondary_missing
                            ],
                            reduced[
                                self.module.secondary_missing + 1 :
                            ],
                        )
                    )
                )
>>>>>>> REPLACE
MECHANISM: Compensated second-head scale-and-two-shear gauge fixing

HYPOTHESIS: Applying the verified Reference Design 2 parameterization will reduce the model from 1,120 to 1,118 learned parameters while retaining accuracy >= 0.99.

INTENDED_EDIT: Fix the second-head projection pivot to a constant scale, eliminate two neighboring projection coefficients, and compensate exactly in the corresponding value basis during initialization and optimization.

EVIDENCE: Reference Design 2 achieved 0.9999 accuracy with 1,118 parameters and 4,999 training steps using this exact additional second-head scale and shear gauge.

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
    """Projection modulo output shifts, three scales, and three shears."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.second_missing = d_model // 2 + 1
        self.register_buffer(
            "fixed_anchor", torch.tensor([0.02, 0.02, 0.0, 0.02])
        )
        self.weight = nn.Parameter(
            torch.empty((d_model - 1) * d_model - 6)
        )
        object.__setattr__(self, "_value_source", None)

    def share_value(self, source: nn.Module) -> None:
        object.__setattr__(self, "_value_source", source)

    def full_weight(self) -> torch.Tensor:
        prefix_count = self.second_missing - 4
        flat = torch.cat(
            (
                self.fixed_anchor[:3],
                self.weight[:prefix_count],
                self.fixed_anchor[3:],
                self.weight.new_zeros(2),
                self.weight[prefix_count:],
                self.weight.new_zeros(self.d_model),
            )
        )
        return flat.view(self.d_model, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                value.weight[head_start].add_(
                    projection._value_init_head_shear
                    * value.weight[head_start + 1]
                )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                head_shear = (
                    full[0, module.second_missing]
                    / full[0, head_start]
                )
                full[:, module.second_missing].sub_(
                    head_shear * full[:, head_start]
                )

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

                reduced = full[:-1].reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            reduced[3:head_start],
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
    prefix_count = second_missing - 4
    full_grad = torch.cat(
        (
            parameter.grad.new_zeros(3),
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

                reduced = full_value[:-1].reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            reduced[3:head_start],
                            reduced[self.module.second_missing + 2 :],
                        )
                    )
                )
>>>>>>> REPLACE
MECHANISM: Rank-conditioned second-head value/projection scale quotient

HYPOTHESIS: Fixing the dynamically selected second-largest second-head projection coefficient at 0.02 will reduce the verified model from 1,119 to 1,118 parameters while retaining at least 99% accuracy, because its multiplicative scale can be absorbed exactly into the corresponding value feature and its rank-conditioned initialization avoids the unstable positional anchors that previously failed.

INTENDED_EDIT: Add a second fixed scale anchor in the permuted second attention head, omit that projection coordinate from learned storage, inversely rescale its value feature during initialization and optimizer recanonicalization, and update full-gradient reconstruction.

EVIDENCE: The largest-magnitude second-head scale quotient achieved 99.99% accuracy at 1,119 parameters after the rank-conditioned second-head shear achieved 100%; unlike the failed third positional scale anchors, the proposed coordinate is explicitly selected as the second-largest magnitude in its head.

<<<<<<< SEARCH
class GaugeFixedProjectionLinear(nn.Linear):
    """Projection modulo output shifts, three scales, and one shear per head."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.head_start = d_model // 2
        self.second_missing = self.head_start + 1
        self.register_buffer(
            "fixed_anchor", torch.tensor([0.02, 0.02, 0.0])
        )
        self.register_buffer(
            "fixed_head_anchor", torch.tensor(0.02)
        )
        self.weight = nn.Parameter(
            torch.empty((d_model - 1) * d_model - 5)
        )
        object.__setattr__(self, "_value_source", None)

    def share_value(self, source: nn.Module) -> None:
        object.__setattr__(self, "_value_source", source)

    def full_weight(self) -> torch.Tensor:
        prefix_count = self.head_start - 3
        flat = torch.cat(
            (
                self.fixed_anchor,
                self.weight[:prefix_count],
                self.fixed_head_anchor.reshape(1),
                self.weight.new_zeros(1),
                self.weight[prefix_count:],
                self.weight.new_zeros(self.d_model),
            )
        )
        return flat.view(self.d_model, self.d_model)
=======
class GaugeFixedProjectionLinear(nn.Linear):
    """Projection modulo output shifts, four scales, and one shear per head."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.head_start = d_model // 2
        self.second_missing = self.head_start + 1
        self.second_scale_anchor = self.second_missing + 1
        self.register_buffer(
            "fixed_anchor", torch.tensor([0.02, 0.02, 0.0])
        )
        self.register_buffer(
            "fixed_head_anchor", torch.tensor(0.02)
        )
        self.register_buffer(
            "fixed_second_head_anchor", torch.tensor(0.02)
        )
        self.weight = nn.Parameter(
            torch.empty((d_model - 1) * d_model - 6)
        )
        object.__setattr__(self, "_value_source", None)

    def share_value(self, source: nn.Module) -> None:
        object.__setattr__(self, "_value_source", source)

    def full_weight(self) -> torch.Tensor:
        prefix_count = self.head_start - 3
        flat = torch.cat(
            (
                self.fixed_anchor,
                self.weight[:prefix_count],
                self.fixed_head_anchor.reshape(1),
                self.weight.new_zeros(1),
                self.fixed_second_head_anchor.reshape(1),
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
                value.weight[head_start].add_(
                    projection._value_init_head_shear
                    * value.weight[head_start + 1]
                )
                value.weight[
                    projection.second_scale_anchor
                ].mul_(
                    projection._value_init_second_head_scale
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
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

                reduced = full[:-1].reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            reduced[3 : module.head_start],
                            reduced[module.second_missing + 1 :],
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
=======
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
                second_head_anchor = full[
                    0, module.second_scale_anchor
                ].clone()
                second_head_scale = (
                    module.fixed_second_head_anchor
                    / second_head_anchor
                )
                full[:, module.second_scale_anchor].mul_(
                    second_head_scale
                )

                reduced = full[:-1].reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            reduced[3 : module.head_start],
                            reduced[module.second_scale_anchor + 1 :],
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
                module._value_init_second_head_scale = (
                    second_head_scale.reciprocal().detach()
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
            parameter.grad[:prefix_count],
            parameter.grad.new_zeros(2),
            parameter.grad[prefix_count:],
=======
            parameter.grad[:prefix_count],
            parameter.grad.new_zeros(3),
            parameter.grad[prefix_count:],
>>>>>>> REPLACE

<<<<<<< SEARCH
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

                reduced = full_value[:-1].reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            reduced[3 : self.module.head_start],
                            reduced[self.module.second_missing + 1 :],
                        )
                    )
                )
=======
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

                second_head_scale = (
                    self.module.fixed_second_head_anchor
                    / full_value[
                        0, self.module.second_scale_anchor
                    ]
                )
                full_value[
                    :, self.module.second_scale_anchor
                ].mul_(second_head_scale)
                self.module._value_source.weight[
                    self.module.second_scale_anchor
                ].div_(second_head_scale)

                reduced = full_value[:-1].reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            reduced[3 : self.module.head_start],
                            reduced[
                                self.module.second_scale_anchor + 1 :
                            ],
                        )
                    )
                )
>>>>>>> REPLACE
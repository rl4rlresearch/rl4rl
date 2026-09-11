MECHANISM: Norm-preserving Givens quotient in the remaining second-head value subspace

HYPOTHESIS: Canonicalizing the two unconstrained second-head features with an orthogonal rotation will reduce the model from 1,119 to 1,118 parameters while retaining at least 99% accuracy, because it removes an exact value/projection basis gauge without the inverse scaling or unbounded shears that caused prior 1,118-parameter failures.

INTENDED_EDIT: Omit one additional second-head projection coordinate, initialize and maintain it at zero with a Givens rotation, and apply the inverse orthogonal transform to the corresponding value features.

EVIDENCE: The conditioned second-head pivot scale quotient achieved 99.99% at 1,119 parameters, while a second scale anchor and an additional shear collapsed at 1,118; a bounded norm-preserving rotation isolates conditioning as the failure source while leaving the verified pivot and shear unchanged.

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
    """Projection modulo shifts, three scales, two shears, and one rotation."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.head_start = d_model // 2
        self.second_missing = self.head_start + 1
        self.second_rotation_missing = self.head_start + 2
        self.register_buffer(
            "fixed_anchor", torch.tensor([0.02, 0.02, 0.0])
        )
        self.register_buffer(
            "fixed_head_anchor", torch.tensor(0.02)
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

                reduced = full[:-1].reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            reduced[3 : module.head_start],
                            reduced[
                                module.second_rotation_missing + 1 :
                            ],
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

                reduced = full_value[:-1].reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            reduced[3 : self.module.head_start],
                            reduced[
                                self.module.second_rotation_missing + 1 :
                            ],
                        )
                    )
                )
>>>>>>> REPLACE
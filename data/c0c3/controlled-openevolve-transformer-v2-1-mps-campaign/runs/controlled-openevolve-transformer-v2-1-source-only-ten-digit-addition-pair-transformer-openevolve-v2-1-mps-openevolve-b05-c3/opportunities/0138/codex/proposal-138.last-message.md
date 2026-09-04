MECHANISM: Norm-conditioned first-head Givens quotient

HYPOTHESIS: Replacing the first head’s raw scalar pivot chart with an orthogonal alignment of features zero and three, followed by the existing 0.02 scale anchor, will reduce the model from 1,117 to 1,116 parameters while retaining at least 99% accuracy because the scale denominator becomes a well-conditioned pair norm.

INTENDED_EDIT: Omit the first row’s fourth projection coefficient, maintain it at zero with a Givens rotation at initialization and after every projection update, and apply the matching rotation to the learned value features before the existing scale and shear transformations.

EVIDENCE: Successive norm-preserving rotations achieved 99.96% at 1,118 parameters and 99.99% at 1,117, while the added rotation-conditioned scale quotient collapsed at 1,116; this tests another orthogonal quotient while retaining the already-qualified scale anchors.

<<<<<<< SEARCH
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
=======
class GaugeFixedProjectionLinear(nn.Linear):
    """Projection modulo shifts, scales, shears, and three rotations."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.head_start = d_model // 2
        self.first_rotation_other = self.head_start - 1
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
            torch.empty((d_model - 1) * d_model - 8)
        )
        object.__setattr__(self, "_value_source", None)

    def share_value(self, source: nn.Module) -> None:
        object.__setattr__(self, "_value_source", source)

    def full_weight(self) -> torch.Tensor:
        nested_prefix_count = (
            self.nested_missing
            - (self.second_rotation_missing + 1)
        )
        flat = torch.cat(
            (
                self.fixed_anchor,
                self.weight.new_zeros(1),
                self.fixed_head_anchor.reshape(1),
                self.weight.new_zeros(2),
                self.weight[:nested_prefix_count],
                self.weight.new_zeros(1),
                self.weight[nested_prefix_count:],
                self.weight.new_zeros(self.d_model),
            )
        )
        return flat.view(self.d_model, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                projection = block.attn.proj
                value = block.attn.value
                value.weight[:2].mul_(
                    projection._value_init_scale[:, None]
                )
                value.weight[0].add_(
                    projection._value_init_shear * value.weight[2]
                )

                head_start = projection.head_start
=======
                projection = block.attn.proj
                value = block.attn.value
                first_other = projection.first_rotation_other
                first_value = torch.stack(
                    (
                        value.weight[0],
                        value.weight[first_other],
                    )
                )
                rotated_first_value = (
                    projection._value_init_first_rotation
                    @ first_value
                )
                value.weight[0].copy_(rotated_first_value[0])
                value.weight[first_other].copy_(
                    rotated_first_value[1]
                )
                value.weight[:2].mul_(
                    projection._value_init_scale[:, None]
                )
                value.weight[0].add_(
                    projection._value_init_shear * value.weight[2]
                )

                head_start = projection.head_start
>>>>>>> REPLACE

<<<<<<< SEARCH
                anchors = full[0, :2].clone()
                scale = module.fixed_anchor[:2] / anchors
                full[:, :2].mul_(scale)
                shear = full[0, 2] / full[0, 0]
                full[:, 2].sub_(shear * full[:, 0])

                head_start = module.head_start
=======
                first_other = module.first_rotation_other
                first_pair = torch.stack(
                    (
                        full[0, 0],
                        full[0, first_other],
                    )
                )
                first_norm = first_pair.norm()
                first_rotation = torch.stack(
                    (
                        first_pair[0] / first_norm,
                        first_pair[1] / first_norm,
                        -first_pair[1] / first_norm,
                        first_pair[0] / first_norm,
                    )
                ).view(2, 2)
                rotated_first = torch.stack(
                    (
                        full[:, 0],
                        full[:, first_other],
                    ),
                    dim=1,
                ) @ first_rotation.transpose(0, 1)
                full[:, 0].copy_(rotated_first[:, 0])
                full[:, first_other].copy_(rotated_first[:, 1])

                anchors = full[0, :2].clone()
                scale = module.fixed_anchor[:2] / anchors
                full[:, :2].mul_(scale)
                shear = full[0, 2] / full[0, 0]
                full[:, 2].sub_(shear * full[:, 0])

                head_start = module.head_start
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
                reduced = full[:-1].reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            reduced[
                                module.second_rotation_missing + 1 :
                                module.nested_missing
                            ],
                            reduced[module.nested_missing + 1 :],
                        )
                    )
                )
                module._value_init_first_rotation = (
                    first_rotation.detach()
                )
                module._value_init_scale = scale.reciprocal().detach()
                module._value_init_shear = shear.detach()
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
def full_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    head_start = d_model // 2
    second_rotation_missing = head_start + 2
    nested_missing = d_model + head_start + 1
    nested_prefix_count = (
        nested_missing - (second_rotation_missing + 1)
    )
    full_grad = torch.cat(
        (
            parameter.grad.new_zeros(
                second_rotation_missing + 1
            ),
            parameter.grad[:nested_prefix_count],
            parameter.grad.new_zeros(1),
            parameter.grad[nested_prefix_count:],
            parameter.grad.new_zeros(d_model),
        )
    ).view(d_model, d_model)
    full_grad[-1] = -full_grad[:-1].sum(dim=0)
    return full_grad
>>>>>>> REPLACE

<<<<<<< SEARCH
                anchors = full_value[0, :2].clone()
                scale = self.module.fixed_anchor[:2] / anchors
                full_value[:, :2].mul_(scale)
                self.module._value_source.weight[:2].div_(
                    scale[:, None]
                )

                shear = full_value[0, 2] / full_value[0, 0]
=======
                first_other = self.module.first_rotation_other
                first_pair = torch.stack(
                    (
                        full_value[0, 0],
                        full_value[0, first_other],
                    )
                )
                first_norm = first_pair.norm()
                first_rotation = torch.stack(
                    (
                        first_pair[0] / first_norm,
                        first_pair[1] / first_norm,
                        -first_pair[1] / first_norm,
                        first_pair[0] / first_norm,
                    )
                ).view(2, 2)
                rotated_first = torch.stack(
                    (
                        full_value[:, 0],
                        full_value[:, first_other],
                    ),
                    dim=1,
                ) @ first_rotation.transpose(0, 1)
                full_value[:, 0].copy_(rotated_first[:, 0])
                full_value[:, first_other].copy_(
                    rotated_first[:, 1]
                )
                first_value = torch.stack(
                    (
                        self.module._value_source.weight[0],
                        self.module._value_source.weight[first_other],
                    )
                )
                rotated_first_value = first_rotation @ first_value
                self.module._value_source.weight[0].copy_(
                    rotated_first_value[0]
                )
                self.module._value_source.weight[first_other].copy_(
                    rotated_first_value[1]
                )

                anchors = full_value[0, :2].clone()
                scale = self.module.fixed_anchor[:2] / anchors
                full_value[:, :2].mul_(scale)
                self.module._value_source.weight[:2].div_(
                    scale[:, None]
                )

                shear = full_value[0, 2] / full_value[0, 0]
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
                reduced = full_value[:-1].reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            reduced[
                                self.module.second_rotation_missing + 1 :
                                self.module.nested_missing
                            ],
                            reduced[self.module.nested_missing + 1 :],
                        )
                    )
                )
>>>>>>> REPLACE
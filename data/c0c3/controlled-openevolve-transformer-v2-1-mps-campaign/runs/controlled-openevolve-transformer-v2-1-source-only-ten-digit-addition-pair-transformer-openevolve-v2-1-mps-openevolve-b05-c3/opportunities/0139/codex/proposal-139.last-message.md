MECHANISM: Five-terminal kernel tie atop the qualified nested orthogonal projection quotient

HYPOTHESIS: Tying a fifth terminal relative-kernel logit while adopting the verified nested Givens quotient will reduce the model from 1,118 to 1,116 parameters and retain at least 99% accuracy, because it preserves the successful bounded projection chart and removes capacity from the independent lag kernel rather than adding another unstable projection scale or shear.

INTENDED_EDIT: Add the qualified second nested rotation and omit its projection coordinate, then shorten the learned relative-bias vector by one while reconstructing five tied terminal logits.

EVIDENCE: The nested orthogonal quotient achieved 99.99% accuracy at 1,117 parameters, while both subsequent 1,116 projection modifications collapsed; this motivates preserving that projection geometry and testing one additional tie in the already four-way-tied relative kernel.

<<<<<<< SEARCH
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
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the four terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 4)
        )
=======
        # Share one discrete kernel at a common temperature, fix the head
        # spacing, and tie the five terminal kernel logits.
        self.relative_bias = nn.Parameter(
            torch.zeros(max_seq_len - 5)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.relative_bias,
                self.relative_bias.new_zeros(4),
=======
                self.relative_bias,
                self.relative_bias.new_zeros(5),
>>>>>>> REPLACE

<<<<<<< SEARCH
                value.weight[
                    rotation_start : rotation_start + 2
                ].copy_(
                    projection._value_init_head_rotation
                    @ rotated_value
                )
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                full[
                    :, rotation_start : rotation_start + 2
                ].copy_(
                    full[
                        :, rotation_start : rotation_start + 2
                    ].clone()
                    @ head_rotation.transpose(0, 1)
                )

                reduced = full[:-1].reshape(-1)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                            reduced[3 : module.head_start],
                            reduced[
                                module.second_rotation_missing + 1 :
                            ],
=======
                            reduced[3 : module.head_start],
                            reduced[
                                module.second_rotation_missing + 1 :
                                module.nested_missing
                            ],
                            reduced[module.nested_missing + 1 :],
>>>>>>> REPLACE

<<<<<<< SEARCH
                module._value_init_head_rotation = (
                    head_rotation.detach()
                )
                if module.bias is not None:
=======
                module._value_init_head_rotation = (
                    head_rotation.detach()
                )
                module._value_init_nested_rotation = (
                    nested_rotation.detach()
                )
                if module.bias is not None:
>>>>>>> REPLACE

<<<<<<< SEARCH
def full_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    head_start = d_model // 2
    prefix_count = head_start - 3
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
=======
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
>>>>>>> REPLACE
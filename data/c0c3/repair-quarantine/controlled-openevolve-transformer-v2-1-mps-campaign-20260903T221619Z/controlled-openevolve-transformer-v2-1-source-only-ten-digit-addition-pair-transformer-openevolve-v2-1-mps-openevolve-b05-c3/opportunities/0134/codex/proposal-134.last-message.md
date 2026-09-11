MECHANISM: Conditioned second-largest-head value/projection scale quotient

HYPOTHESIS: Fixing the dynamically selected second-largest second-head projection coefficient at 0.02 will reduce the qualified model from 1,119 to 1,118 parameters while retaining at least 99% accuracy, because the exact scale is absorbed into its corresponding value feature and dynamic ordering avoids the unstable near-zero coefficients implicated in earlier scale-anchor failures.

INTENDED_EDIT: Add a second fixed scale anchor in the dynamically ordered second head, omit that projection coordinate from learned storage, compensate its value feature during initialization and every projection update, and update reduced-gradient reconstruction.

EVIDENCE: The dynamically selected largest second-head scale anchor achieved 99.99% accuracy with 1,119 parameters, while prior additional anchors on an unstable first-head coefficient collapsed; selecting the second-largest coefficient preserves the successful conditioning strategy while testing another exact factorization gauge.

<<<<<<< SEARCH
class GaugeFixedProjectionLinear(nn.Linear):
    """Projection modulo output shifts, three scales, and one shear per head."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.second_missing = d_model // 2 + 1
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
        prefix_count = self.second_missing - 4
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
        self.second_missing = d_model // 2 + 1
        self.register_buffer(
            "fixed_anchor", torch.tensor([0.02, 0.02, 0.0])
        )
        self.register_buffer(
            "fixed_head_anchor", torch.tensor([0.02, 0.02])
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
                self.fixed_anchor,
                self.weight[:prefix_count],
                self.fixed_head_anchor[:1],
                self.weight.new_zeros(1),
                self.fixed_head_anchor[1:],
                self.weight[prefix_count:],
                self.weight.new_zeros(self.d_model),
            )
        )
        return flat.view(self.d_model, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                value.weight[head_start].mul_(
                    projection._value_init_head_scale
                )
=======
                value.weight[
                    head_start : projection.second_missing + 2 : 2
                ].mul_(
                    projection._value_init_head_scale[:, None]
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                head_scale = (
                    module.fixed_head_anchor
                    / full[0, head_start]
                )
                full[:, head_start].mul_(head_scale)

                reduced = full[:-1].reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            reduced[3:head_start],
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
                module._value_init_head_scale = (
                    head_scale.reciprocal().detach()
                )
=======
                head_scale_columns = slice(
                    head_start, module.second_missing + 2, 2
                )
                head_scale = (
                    module.fixed_head_anchor
                    / full[0, head_scale_columns]
                )
                full[:, head_scale_columns].mul_(head_scale)

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
                module._value_init_head_shear = head_shear.detach()
                module._value_init_head_scale = (
                    head_scale.reciprocal().detach()
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
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
            parameter.grad.new_zeros(2),
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
                head_scale = (
                    self.module.fixed_head_anchor
                    / full_value[0, head_start]
                )
                full_value[:, head_start].mul_(head_scale)
                self.module._value_source.weight[head_start].div_(
                    head_scale
                )

                reduced = full_value[:-1].reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            reduced[3:head_start],
                            reduced[self.module.second_missing + 1 :],
                        )
                    )
                )
=======
                head_scale_columns = slice(
                    head_start, self.module.second_missing + 2, 2
                )
                head_scale = (
                    self.module.fixed_head_anchor
                    / full_value[0, head_scale_columns]
                )
                full_value[:, head_scale_columns].mul_(head_scale)
                self.module._value_source.weight[
                    head_scale_columns
                ].div_(head_scale[:, None])

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
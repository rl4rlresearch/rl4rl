MECHANISM: Independent-anchor within-head value-basis shear quotient

HYPOTHESIS: Eliminating the fourth projection coefficient through a feature-3-to-feature-1 shear will reduce the qualified 1,121-parameter model to 1,120 parameters while retaining at least 99% accuracy, because it uses the distinct fixed second anchor and exactly preserves the learned attention function.

INTENDED_EDIT: Extend the two-scale, one-shear projection chart with a second zero anchor, compensate through value feature one at initialization and after projection updates, and reconstruct the additional omitted projection gradient.

EVIDENCE: The first stable shear achieved 99.98% accuracy at 1,121 parameters, while a second shear sharing value feature zero collapsed; the independent-anchor variant was not tested because its patch search matched multiple locations.

<<<<<<< SEARCH
class GaugeFixedProjectionLinear(nn.Linear):
    """Projection modulo output shifts, two scales, and one value shear."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.register_buffer(
            "fixed_anchor", torch.tensor([0.02, 0.02, 0.0])
        )
        self.weight = nn.Parameter(
            torch.empty((d_model - 1) * d_model - 3)
        )
=======
class GaugeFixedProjectionLinear(nn.Linear):
    """Projection modulo output shifts, two scales, and two value shears."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.register_buffer(
            "fixed_anchor", torch.tensor([0.02, 0.02, 0.0, 0.0])
        )
        self.weight = nn.Parameter(
            torch.empty((d_model - 1) * d_model - 4)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        with torch.no_grad():
            for block in self.blocks:
                block.attn.value.weight[:2].mul_(
                    block.attn.proj._value_init_scale[:, None]
                )
                block.attn.value.weight[0].add_(
                    block.attn.proj._value_init_shear
                    * block.attn.value.weight[2]
                )
=======
        with torch.no_grad():
            for block in self.blocks:
                block.attn.value.weight[:2].mul_(
                    block.attn.proj._value_init_scale[:, None]
                )
                block.attn.value.weight[0].add_(
                    block.attn.proj._value_init_shear[0]
                    * block.attn.value.weight[2]
                )
                block.attn.value.weight[1].add_(
                    block.attn.proj._value_init_shear[1]
                    * block.attn.value.weight[3]
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                anchors = full[0, :2].clone()
                scale = module.fixed_anchor[:2] / anchors
                full[:, :2].mul_(scale)
                shear = full[0, 2] / full[0, 0]
                full[:, 2].sub_(shear * full[:, 0])
                module.weight.copy_(full[:-1].reshape(-1)[3:])
                module._value_init_scale = scale.reciprocal().detach()
                module._value_init_shear = shear.detach()
                if module.bias is not None:
=======
                anchors = full[0, :2].clone()
                scale = module.fixed_anchor[:2] / anchors
                full[:, :2].mul_(scale)
                shear = torch.stack(
                    (
                        full[0, 2] / full[0, 0],
                        full[0, 3] / full[0, 1],
                    )
                )
                full[:, 2].sub_(shear[0] * full[:, 0])
                full[:, 3].sub_(shear[1] * full[:, 1])
                module.weight.copy_(full[:-1].reshape(-1)[4:])
                module._value_init_scale = scale.reciprocal().detach()
                module._value_init_shear = shear.detach()
                if module.bias is not None:
>>>>>>> REPLACE

<<<<<<< SEARCH
def full_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    full_grad = torch.cat(
        (
            parameter.grad.new_zeros(3),
            parameter.grad,
            parameter.grad.new_zeros(d_model),
        )
    ).view(d_model, d_model)
=======
def full_projection_gradient(
    parameter: torch.nn.Parameter,
    d_model: int,
) -> torch.Tensor:
    full_grad = torch.cat(
        (
            parameter.grad.new_zeros(4),
            parameter.grad,
            parameter.grad.new_zeros(d_model),
        )
    ).view(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
                parameter.copy_(full_value[:-1].reshape(-1)[3:])
=======
                anchors = full_value[0, :2].clone()
                scale = self.module.fixed_anchor[:2] / anchors
                full_value[:, :2].mul_(scale)
                self.module._value_source.weight[:2].div_(
                    scale[:, None]
                )

                shear_0 = full_value[0, 2] / full_value[0, 0]
                full_value[:, 2].sub_(
                    shear_0 * full_value[:, 0]
                )
                self.module._value_source.weight[0].add_(
                    shear_0 * self.module._value_source.weight[2]
                )

                shear_1 = full_value[0, 3] / full_value[0, 1]
                full_value[:, 3].sub_(
                    shear_1 * full_value[:, 1]
                )
                self.module._value_source.weight[1].add_(
                    shear_1 * self.module._value_source.weight[3]
                )
                parameter.copy_(full_value[:-1].reshape(-1)[4:])
>>>>>>> REPLACE
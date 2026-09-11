MECHANISM: Second stable within-head value-basis shear quotient

HYPOTHESIS: Extending the qualified two-scale, one-shear 1,121-parameter design with a second shear anchored by the fixed nonzero first projection coefficient will reduce it to 1,120 parameters while retaining at least 99% accuracy, because the transformation exactly preserves the learned attention function without dividing by another unstable coefficient.

INTENDED_EDIT: Adopt the qualified two projection-scale anchors and first within-head shear, then eliminate the fourth projection coefficient by shearing value feature three into feature zero during initialization and every projection optimizer step.

EVIDENCE: The first within-head shear achieved 99.98% accuracy with 1,121 parameters, while attempts to use the third coefficient as a scale anchor failed at 0.01% and 42.48%; another shear through the already-fixed 0.02 anchor tests the same successful, well-conditioned symmetry removal.

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
        object.__setattr__(self, "_value_source", None)
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
                block.attn.value.weight[:2].mul_(
                    block.attn.proj._value_init_scale[:, None]
                )
                block.attn.value.weight[0].add_(
                    block.attn.proj._value_init_shear
                    @ block.attn.value.weight[2:4]
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                anchor = full[0, 0].clone()
                scale = module.fixed_anchor / anchor
                full[:, 0].mul_(scale)
                module.weight.copy_(full[:-1].reshape(-1)[1:])
                module._value_init_scale = scale.reciprocal().detach()
=======
                anchors = full[0, :2].clone()
                scale = module.fixed_anchor[:2] / anchors
                full[:, :2].mul_(scale)
                shear = full[0, 2:4] / full[0, 0]
                full[:, 2:4].sub_(full[:, 0:1] * shear)
                module.weight.copy_(full[:-1].reshape(-1)[4:])
                module._value_init_scale = scale.reciprocal().detach()
                module._value_init_shear = shear.detach()
>>>>>>> REPLACE

<<<<<<< SEARCH
    full_grad = torch.cat(
        (
            parameter.grad.new_zeros(1),
            parameter.grad,
            parameter.grad.new_zeros(d_model),
        )
    ).view(d_model, d_model)
=======
    full_grad = torch.cat(
        (
            parameter.grad.new_zeros(4),
            parameter.grad,
            parameter.grad.new_zeros(d_model),
        )
    ).view(d_model, d_model)
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

                shear = full_value[0, 2:4] / full_value[0, 0]
                full_value[:, 2:4].sub_(
                    full_value[:, 0:1] * shear
                )
                self.module._value_source.weight[0].add_(
                    shear @ self.module._value_source.weight[2:4]
                )
                parameter.copy_(full_value[:-1].reshape(-1)[4:])
>>>>>>> REPLACE
MECHANISM: Within-head value-basis shear quotient

HYPOTHESIS: Fixing one attention-projection coefficient to zero through a triangular value-basis shear will reduce the qualified model from 1,122 to 1,121 parameters while retaining at least 99% accuracy, because it preserves the learned attention function and avoids dividing by the unstable third coefficient that caused both three-scale-anchor failures.

INTENDED_EDIT: Preserve the two qualified value-scale anchors, remove the third projection coefficient, and absorb it into a same-head shear between value features zero and two during initialization and optimizer updates.

EVIDENCE: The two-feature scale quotient achieved 99.98% accuracy at 1,122 parameters, while third scale anchors achieved only 0.01% and 42.48%; the proposed shear instead uses the already-fixed nonzero first anchor as its stable denominator and removes an exact within-head factorization gauge.

<<<<<<< SEARCH
class GaugeFixedProjectionLinear(nn.Linear):
    """Attention projection modulo output shifts and two value-scale gauges."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.register_buffer("fixed_anchor", torch.full((2,), 0.02))
        self.weight = nn.Parameter(
            torch.empty((d_model - 1) * d_model - 2)
        )
        object.__setattr__(self, "_value_source", None)

    def share_value(self, source: nn.Module) -> None:
        object.__setattr__(self, "_value_source", source)

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.fixed_anchor,
                self.weight,
                self.weight.new_zeros(self.d_model),
            )
        )
        return flat.view(self.d_model, self.d_model)
=======
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
        object.__setattr__(self, "_value_source", None)

    def share_value(self, source: nn.Module) -> None:
        object.__setattr__(self, "_value_source", source)

    def full_weight(self) -> torch.Tensor:
        flat = torch.cat(
            (
                self.fixed_anchor,
                self.weight,
                self.weight.new_zeros(self.d_model),
            )
        )
        return flat.view(self.d_model, self.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        with torch.no_grad():
            for block in self.blocks:
                block.attn.value.weight[:2].mul_(
                    block.attn.proj._value_init_scale[:, None]
                )
=======
        with torch.no_grad():
            for block in self.blocks:
                block.attn.value.weight[:2].mul_(
                    block.attn.proj._value_init_scale[:, None]
                )
                block.attn.value.weight[0].add_(
                    block.attn.proj._value_init_shear
                    * block.attn.value.weight[2]
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
                anchors = full[0, :2].clone()
                scale = module.fixed_anchor / anchors
                full[:, :2].mul_(scale)
                module.weight.copy_(full[:-1].reshape(-1)[2:])
                module._value_init_scale = scale.reciprocal().detach()
                if module.bias is not None:
=======
                anchors = full[0, :2].clone()
                scale = module.fixed_anchor[:2] / anchors
                full[:, :2].mul_(scale)
                shear = full[0, 2] / full[0, 0]
                full[:, 2].sub_(shear * full[:, 0])
                module.weight.copy_(full[:-1].reshape(-1)[3:])
                module._value_init_scale = scale.reciprocal().detach()
                module._value_init_shear = shear.detach()
                if module.bias is not None:
>>>>>>> REPLACE

<<<<<<< SEARCH
    full_grad = torch.cat(
        (
            parameter.grad.new_zeros(2),
            parameter.grad,
            parameter.grad.new_zeros(d_model),
        )
    ).view(d_model, d_model)
=======
    full_grad = torch.cat(
        (
            parameter.grad.new_zeros(3),
            parameter.grad,
            parameter.grad.new_zeros(d_model),
        )
    ).view(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                anchors = full_value[0, :2].clone()
                scale = self.module.fixed_anchor / anchors
                full_value[:, :2].mul_(scale)
                self.module._value_source.weight[:2].div_(
                    scale[:, None]
                )
                parameter.copy_(full_value[:-1].reshape(-1)[2:])
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
                parameter.copy_(full_value[:-1].reshape(-1)[3:])
>>>>>>> REPLACE
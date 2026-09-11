MECHANISM: Rank-conditioned dual shear in the second attention head

HYPOTHESIS: Shearing the second-smallest second-head projection coordinate through the verified largest-magnitude 0.02 pivot will reduce the qualified 1,119-parameter model to 1,118 parameters while retaining at least 99% accuracy, because it exactly preserves the learned attention function without the destabilizing inverse rescaling used by the failed second-largest scale quotient.

INTENDED_EDIT: Adopt the qualified three-scale, two-shear projection chart, then dynamically place the two smallest second-head coordinates beside its largest pivot and eliminate both through bounded value-basis shears.

EVIDENCE: The largest-magnitude second-head pivot scale quotient reached 99.99% accuracy at 1,119 parameters, whereas scaling the second-largest coordinate collapsed to 0.17%; a shear through the already conditioned pivot tests another exact gauge without independently normalizing that coordinate.

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
    """Projection modulo output shifts, three scales, and three shears."""

    def __init__(self, d_model: int):
        super().__init__(d_model, d_model)
        self.d_model = d_model
        self.head_start = d_model // 2
        self.second_missing = self.head_start + 1
        self.third_missing = self.head_start + 2
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
        self.apply(self._init_weights)
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
        self.apply(self._init_weights)
        with torch.no_grad():
            for block in self.blocks:
                projection = block.attn.proj
                value = block.attn.value
                value.weight[:2].mul_(
                    projection._value_init_scale[:, None]
                )
                value.weight[0].add_(
                    projection._value_init_shear * value.weight[2]
                )

                head_start = projection.head_start
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
                    projection._value_init_head_shear
                    * value.weight[projection.second_missing]
                )
                value.weight[head_start].add_(
                    projection._value_init_head_second_shear
                    * value.weight[projection.third_missing]
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedProjectionLinear):
            d_model = module.d_model
            full = module.weight.new_empty(d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1].clone()
                full.sub_(omitted)
                full[-1].zero_()
                anchors = full[0, :2].clone()
                scale = module.fixed_anchor[:2] / anchors
                full[:, :2].mul_(scale)
                shear = full[0, 2] / full[0, 0]
                full[:, 2].sub_(shear * full[:, 0])
                module.weight.copy_(full[:-1].reshape(-1)[3:])
                module._value_init_scale = scale.reciprocal().detach()
                module._value_init_shear = shear.detach()
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
=======
        elif isinstance(module, GaugeFixedProjectionLinear):
            d_model = module.d_model
            full = module.weight.new_empty(d_model, d_model)
            nn.init.normal_(full, mean=0.0, std=0.02)
            with torch.no_grad():
                omitted = full[-1].clone()
                full.sub_(omitted)
                full[-1].zero_()

                anchors = full[0, :2].clone()
                scale = module.fixed_anchor[:2] / anchors
                full[:, :2].mul_(scale)
                shear = full[0, 2] / full[0, 0]
                full[:, 2].sub_(shear * full[:, 0])

                head_start = module.head_start
                head_order = torch.argsort(
                    full[0, head_start:].abs(),
                    descending=True,
                )
                head_permutation = torch.cat(
                    (
                        head_order[:1],
                        head_order[-1:],
                        head_order[-2:-1],
                        head_order[1:-2],
                    )
                )
                permuted_head = full[:, head_start:].clone()[
                    :, head_permutation
                ]
                full[:, head_start:].copy_(permuted_head)

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
                head_second_shear = (
                    full[0, module.third_missing]
                    / full[0, head_start]
                )
                full[:, module.third_missing].sub_(
                    head_second_shear * full[:, head_start]
                )

                reduced = full[:-1].reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            reduced[3 : module.head_start],
                            reduced[module.third_missing + 1 :],
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
                module._value_init_head_second_shear = (
                    head_second_shear.detach()
                )
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
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
    full_grad[-1] = -full_grad[:-1].sum(dim=0)
    return full_grad
=======
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

                shear = full_value[0, 2] / full_value[0, 0]
                full_value[:, 2].sub_(
                    shear * full_value[:, 0]
                )
                self.module._value_source.weight[0].add_(
                    shear * self.module._value_source.weight[2]
                )

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

                head_second_shear = (
                    full_value[0, self.module.third_missing]
                    / full_value[0, head_start]
                )
                full_value[:, self.module.third_missing].sub_(
                    head_second_shear * full_value[:, head_start]
                )
                self.module._value_source.weight[head_start].add_(
                    head_second_shear
                    * self.module._value_source.weight[
                        self.module.third_missing
                    ]
                )

                reduced = full_value[:-1].reshape(-1)
                parameter.copy_(
                    torch.cat(
                        (
                            reduced[3 : self.module.head_start],
                            reduced[self.module.third_missing + 1 :],
                        )
                    )
                )
>>>>>>> REPLACE
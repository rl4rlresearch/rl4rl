MECHANISM: First pre-MLP zero-mean row gauge with ambient factorized AdamW

HYPOTHESIS: A 1,378-parameter model will retain at least 99% accuracy because the qualified 1,379-parameter design reached 99.67%, while one common component of an `fc1` weight row is functionally invisible after parameter-free LayerNorm and its omitted ambient coordinate is preserved during optimization.

INTENDED_EDIT: Reproduce the qualified complete terminal gauge and four-column attention-output gauge, then remove one parameter from the first `fc1` row while maintaining its omitted offset through gradient clipping, absorbed-scale AdamW updates, and initialization.

EVIDENCE: Reference Design 3 achieved 99.67% accuracy at 1,379 parameters. The failed fifth attention-output gauge motivates testing a distinct exact LayerNorm symmetry rather than extending that attention gauge again.

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and eleven output-shift column gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(11)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 11)
        )
        self.bias = nn.Parameter(torch.empty(out_features - 1))
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and every output-shift gauge removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(in_features)
            ]
        )
        self.bias = nn.Parameter(torch.empty(out_features - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
    @torch.no_grad()
    def reset_parameters(self) -> None:
        raw_weight = self.weight_rest.new_empty(
            self.out_features, self.in_features
        )
        nn.init.kaiming_uniform_(raw_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(raw_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 11:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
    @torch.no_grad()
    def reset_parameters(self) -> None:
        raw_weight = self.bias.new_empty(
            self.out_features, self.in_features
        )
        nn.init.kaiming_uniform_(raw_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(raw_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
        weight = torch.cat(
            (
                torch.stack(full_weight_prefix, dim=1),
                self.weight_rest,
            ),
            dim=1,
        )
        return F.linear(x, weight, full_bias)


class GaugeFixedAttentionProjection(nn.Module):
=======
        weight = torch.stack(full_weight_prefix, dim=1)
        return F.linear(x, weight, full_bias)


class GaugeFixedAttentionProjection(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with bias and three weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(3)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 3)
        )
=======
class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with bias and four weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(4)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 4)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest.copy_(raw_weight[:, 3:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
=======
        self.weight_rest.copy_(raw_weight[:, 4:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
>>>>>>> REPLACE

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
=======
class GaugeFixedFirstRowLinear(nn.Module):
    """Linear layer for zero-mean inputs with one row-shift gauge removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_first = nn.Parameter(torch.empty(in_features - 1))
        self.weight_rest = nn.Parameter(
            torch.empty(out_features - 1, in_features)
        )
        self.bias = nn.Parameter(torch.empty(out_features))
        self.register_buffer(
            "initial_offset", torch.zeros(()), persistent=False
        )
        self.full_weight = None
        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self) -> None:
        raw_weight = self.weight_rest.new_empty(
            self.out_features, self.in_features
        )
        nn.init.kaiming_uniform_(raw_weight, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(raw_weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        raw_bias = self.bias.new_empty(self.out_features)
        nn.init.uniform_(raw_bias, -bound, bound)
        offset = raw_weight[0, -1]
        self.weight_first.copy_(raw_weight[0, :-1] - offset)
        self.weight_rest.copy_(raw_weight[1:])
        self.bias.copy_(raw_bias)
        self.initial_offset.copy_(offset)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first = torch.cat(
            (self.weight_first, self.weight_first.new_zeros(1))
        )
        weight = torch.cat(
            (first.unsqueeze(0), self.weight_rest), dim=0
        )
        if torch.is_grad_enabled():
            weight.retain_grad()
            self.full_weight = weight
        return F.linear(x, weight, self.bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = GaugeFixedFirstRowLinear(d_model, d_ff)
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedAttentionProjection):
            with torch.no_grad():
                raw_weight = module.weight_rest.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 3:])
                nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedTerminalLinear):
            with torch.no_grad():
                raw_weight = module.weight_rest.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 11:])
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
        elif isinstance(module, GaugeFixedAttentionProjection):
            with torch.no_grad():
                raw_weight = module.weight_rest.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 4:])
                nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedTerminalLinear):
            with torch.no_grad():
                raw_weight = module.bias.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                nn.init.zeros_(module.bias)
        elif isinstance(module, GaugeFixedFirstRowLinear):
            with torch.no_grad():
                raw_weight = module.weight_rest.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                offset = raw_weight[0, -1]
                module.weight_first.copy_(
                    raw_weight[0, :-1] - offset
                )
                module.weight_rest.copy_(raw_weight[1:])
                module.initial_offset.copy_(offset)
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
>>>>>>> REPLACE

<<<<<<< SEARCH
    # All eight ln2 scales live only in optimizer-coordinate state; fc1
    # stores and uses their products with the ambient weight columns.
    absorbed_scales = [
        torch.ones(8, device=device, dtype=blk.mlp.fc1.weight.dtype)
        for blk in model.blocks
    ]
    absorbed_weight_m = [
        torch.zeros_like(blk.mlp.fc1.weight[:, -8:])
        for blk in model.blocks
    ]
=======
    # All eight ln2 scales and the omitted common component of fc1's first
    # row live in optimizer-coordinate state.
    absorbed_scales = [
        torch.ones(
            8, device=device, dtype=blk.mlp.fc1.weight_rest.dtype
        )
        for blk in model.blocks
    ]
    fc1_gauge_offsets = [
        blk.mlp.fc1.initial_offset.detach().clone()
        for blk in model.blocks
    ]
    absorbed_weight_m = [
        torch.zeros(
            blk.mlp.fc1.out_features,
            blk.mlp.fc1.in_features,
            device=device,
            dtype=blk.mlp.fc1.weight_rest.dtype,
        )
        for blk in model.blocks
    ]
>>>>>>> REPLACE

<<<<<<< SEARCH
        absorbed_grads = []
        for blk, virtual_scale in zip(model.blocks, absorbed_scales):
            effective_grad = (
                blk.mlp.fc1.weight.grad[:, -8:].detach().clone()
            )
            virtual_weight = (
                blk.mlp.fc1.weight[:, -8:].detach()
                / virtual_scale.unsqueeze(0)
            )
            ambient_weight_grad = (
                effective_grad * virtual_scale.unsqueeze(0)
            )
            ambient_scale_grad = (
                effective_grad * virtual_weight
            ).sum(dim=0)
            absorbed_grads.append(
                (
                    effective_grad,
                    virtual_weight,
                    ambient_weight_grad,
                    ambient_scale_grad,
                )
            )
=======
        absorbed_grads = []
        for blk, virtual_scale, gauge_offset in zip(
            model.blocks, absorbed_scales, fc1_gauge_offsets
        ):
            effective_grad = (
                blk.mlp.fc1.full_weight.grad.detach().clone()
            )
            effective_weight = (
                blk.mlp.fc1.full_weight.detach().clone()
            )
            effective_weight[0].add_(gauge_offset)
            virtual_weight = (
                effective_weight / virtual_scale.unsqueeze(0)
            )
            ambient_weight_grad = (
                effective_grad * virtual_scale.unsqueeze(0)
            )
            ambient_scale_grad = (
                effective_grad * virtual_weight
            ).sum(dim=0)
            absorbed_grads.append(
                (
                    effective_grad,
                    virtual_weight,
                    ambient_weight_grad,
                    ambient_scale_grad,
                )
            )
>>>>>>> REPLACE

<<<<<<< SEARCH
            for (
                effective_grad,
                _,
                ambient_weight_grad,
                ambient_scale_grad,
            ) in absorbed_grads:
                grad_sq = (
                    grad_sq
                    - effective_grad.float().square().sum()
                    + ambient_weight_grad.float().square().sum()
                    + ambient_scale_grad.float().square().sum()
                )
=======
            for (
                effective_grad,
                _,
                ambient_weight_grad,
                ambient_scale_grad,
            ) in absorbed_grads:
                grad_sq = (
                    grad_sq
                    - effective_grad.float().square().sum()
                    + effective_grad[0, -1].float().square()
                    + ambient_weight_grad.float().square().sum()
                    + ambient_scale_grad.float().square().sum()
                )
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -2:].zero_()
            blk.attn.k_proj.weight.grad[:, -2:].zero_()
            blk.attn.v_proj.weight.grad[:, -2:].zero_()
            blk.mlp.fc1.weight.grad[:, -8:].zero_()
=======
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -2:].zero_()
            blk.attn.k_proj.weight.grad[:, -2:].zero_()
            blk.attn.v_proj.weight.grad[:, -2:].zero_()
            blk.mlp.fc1.weight_first.grad.zero_()
            blk.mlp.fc1.weight_rest.grad.zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
                blk.mlp.fc1.weight[:, -8:].copy_(
                    virtual_weight * virtual_scale.unsqueeze(0)
                )

        gauge_step += 1
=======
                effective_weight = (
                    virtual_weight * virtual_scale.unsqueeze(0)
                )
                gauge_offset = effective_weight[0, -1].clone()
                fc1_gauge_offsets[i].copy_(gauge_offset)
                blk.mlp.fc1.weight_first.copy_(
                    effective_weight[0, :-1] - gauge_offset
                )
                blk.mlp.fc1.weight_rest.copy_(
                    effective_weight[1:]
                )

        gauge_step += 1
>>>>>>> REPLACE
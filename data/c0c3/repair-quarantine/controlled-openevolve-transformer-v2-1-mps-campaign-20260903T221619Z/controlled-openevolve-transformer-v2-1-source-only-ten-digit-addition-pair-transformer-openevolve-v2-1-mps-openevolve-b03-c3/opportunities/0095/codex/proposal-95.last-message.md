MECHANISM: Balanced fifth attention-output scalar-shift gauge

HYPOTHESIS: A 1,377-parameter model will retain at least 99% accuracy because the qualified 1,378-parameter design reached 99.39%, while gauge-fixing the untested attention column five with a zero-mean representative minimizes the numerically irrelevant common output shift that may have destabilized prior fifth-gauge trials.

INTENDED_EDIT: Reproduce the qualified complete terminal gauge, four-column attention gauge, and three-scale `ln1` absorption, then remove one additional parameter by reconstructing attention column five in a centered scalar-shift gauge while retaining full ambient-gradient clipping and AdamW updates.

EVIDENCE: Reference Design 3 achieved 99.39% at 1,378 parameters. Prior fifth-gauge attempts using last-coordinate-zero representatives collapsed on columns four and seven, motivating a centered representative on the untested interior column five to distinguish numerical gauge conditioning from a true capacity limit.

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
        self.full_weight_prefix = None
        self.full_bias = None
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
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 11:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight_prefix = [
            torch.cat((stored, stored.new_zeros(1)))
            for stored in self.weight_prefix
        ]
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            for full_weight in full_weight_prefix:
                full_weight.retain_grad()
            full_bias.retain_grad()
            self.full_weight_prefix = full_weight_prefix
            self.full_bias = full_bias
        weight = torch.cat(
            (
                torch.stack(full_weight_prefix, dim=1),
                self.weight_rest,
            ),
            dim=1,
        )
        return F.linear(x, weight, full_bias)
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with every output-shift gauge removed."""

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
        self.full_weight_prefix = None
        self.full_bias = None
        self.reset_parameters()

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight_prefix = [
            torch.cat((stored, stored.new_zeros(1)))
            for stored in self.weight_prefix
        ]
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            for full_weight in full_weight_prefix:
                full_weight.retain_grad()
            full_bias.retain_grad()
            self.full_weight_prefix = full_weight_prefix
            self.full_bias = full_bias
        weight = torch.stack(full_weight_prefix, dim=1)
        return F.linear(x, weight, full_bias)
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
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_weight_prefix = None
        self.full_bias = None
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
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 3:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight_prefix = [
            torch.cat((stored, stored.new_zeros(1)))
            for stored in self.weight_prefix
        ]
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            for full_weight in full_weight_prefix:
                full_weight.retain_grad()
            full_bias.retain_grad()
            self.full_weight_prefix = full_weight_prefix
            self.full_bias = full_bias
        weight = torch.cat(
            (
                torch.stack(full_weight_prefix, dim=1),
                self.weight_rest,
            ),
            dim=1,
        )
        return F.linear(x, weight, full_bias)
=======
class GaugeFixedAttentionProjection(nn.Module):
    """Projection with four terminal-zero and one centered column gauges."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gauge_columns = (0, 1, 2, 3, 5)
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in self.gauge_columns
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(
                out_features, in_features - len(self.gauge_columns)
            )
        )
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_weight_prefix = None
        self.full_bias = None
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
        for column, stored in zip(
            self.gauge_columns, self.weight_prefix
        ):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        rest_columns = [
            column
            for column in range(self.in_features)
            if column not in self.gauge_columns
        ]
        self.weight_rest.copy_(raw_weight[:, rest_columns])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight_prefix = [
            torch.cat((stored, stored.new_zeros(1)))
            for stored in self.weight_prefix[:-1]
        ]
        balanced = self.weight_prefix[-1]
        offset = (
            -balanced.sum() / self.out_features
        ).detach()
        full_weight_prefix.append(
            torch.cat(
                (
                    balanced + offset,
                    offset.unsqueeze(0),
                )
            )
        )
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            for full_weight in full_weight_prefix:
                full_weight.retain_grad()
            full_bias.retain_grad()
            self.full_weight_prefix = full_weight_prefix
            self.full_bias = full_bias

        columns = []
        gauge_index = 0
        rest_index = 0
        for column in range(self.in_features):
            if column in self.gauge_columns:
                columns.append(full_weight_prefix[gauge_index])
                gauge_index += 1
            else:
                columns.append(self.weight_rest[:, rest_index])
                rest_index += 1
        weight = torch.stack(columns, dim=1)
        return F.linear(x, weight, full_bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
class TwoAbsorbedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with its final two scales absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(2)))
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )
=======
class ThreeAbsorbedScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with its final three scales absorbed downstream."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(3)))
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = TwoAbsorbedScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = ThreeAbsorbedScaleLayerNorm(cfg.d_model)
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
=======
        elif isinstance(module, GaugeFixedAttentionProjection):
            with torch.no_grad():
                raw_weight = module.weight_rest.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(raw_weight, mean=0.0, std=0.02)
                for column, stored in zip(
                    module.gauge_columns, module.weight_prefix
                ):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                rest_columns = [
                    column
                    for column in range(module.in_features)
                    if column not in module.gauge_columns
                ]
                module.weight_rest.copy_(raw_weight[:, rest_columns])
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
>>>>>>> REPLACE

<<<<<<< SEARCH
    # The final two ln1 scales live only in optimizer-coordinate state;
    # q, k, and v store their products with the corresponding input columns.
    attention_scales = [
        torch.ones(
            2, device=device, dtype=blk.attn.q_proj.weight.dtype
        )
        for blk in model.blocks
    ]
    attention_weight_m = [
        torch.zeros_like(
            torch.cat(
                (
                    blk.attn.q_proj.weight[:, -2:],
                    blk.attn.k_proj.weight[:, -2:],
                    blk.attn.v_proj.weight[:, -2:],
                ),
                dim=0,
            )
        )
=======
    # The final three ln1 scales live only in optimizer-coordinate state;
    # q, k, and v store their products with the corresponding input columns.
    attention_scales = [
        torch.ones(
            3, device=device, dtype=blk.attn.q_proj.weight.dtype
        )
        for blk in model.blocks
    ]
    attention_weight_m = [
        torch.zeros_like(
            torch.cat(
                (
                    blk.attn.q_proj.weight[:, -3:],
                    blk.attn.k_proj.weight[:, -3:],
                    blk.attn.v_proj.weight[:, -3:],
                ),
                dim=0,
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
            effective_grad = torch.cat(
                (
                    blk.attn.q_proj.weight.grad[:, -2:],
                    blk.attn.k_proj.weight.grad[:, -2:],
                    blk.attn.v_proj.weight.grad[:, -2:],
                ),
                dim=0,
            ).detach().clone()
            effective_weight = torch.cat(
                (
                    blk.attn.q_proj.weight[:, -2:],
                    blk.attn.k_proj.weight[:, -2:],
                    blk.attn.v_proj.weight[:, -2:],
                ),
                dim=0,
            ).detach()
=======
            effective_grad = torch.cat(
                (
                    blk.attn.q_proj.weight.grad[:, -3:],
                    blk.attn.k_proj.weight.grad[:, -3:],
                    blk.attn.v_proj.weight.grad[:, -3:],
                ),
                dim=0,
            ).detach().clone()
            effective_weight = torch.cat(
                (
                    blk.attn.q_proj.weight[:, -3:],
                    blk.attn.k_proj.weight[:, -3:],
                    blk.attn.v_proj.weight[:, -3:],
                ),
                dim=0,
            ).detach()
>>>>>>> REPLACE

<<<<<<< SEARCH
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -2:].zero_()
            blk.attn.k_proj.weight.grad[:, -2:].zero_()
            blk.attn.v_proj.weight.grad[:, -2:].zero_()
            blk.mlp.fc1.weight.grad[:, -8:].zero_()
=======
        for blk in model.blocks:
            blk.attn.q_proj.weight.grad[:, -3:].zero_()
            blk.attn.k_proj.weight.grad[:, -3:].zero_()
            blk.attn.v_proj.weight.grad[:, -3:].zero_()
            blk.mlp.fc1.weight.grad[:, -8:].zero_()
>>>>>>> REPLACE

<<<<<<< SEARCH
                blk.attn.q_proj.weight[:, -2:].copy_(
                    effective_weight[:q_end]
                )
                blk.attn.k_proj.weight[:, -2:].copy_(
                    effective_weight[q_end:k_end]
                )
                blk.attn.v_proj.weight[:, -2:].copy_(
                    effective_weight[k_end:]
                )
=======
                blk.attn.q_proj.weight[:, -3:].copy_(
                    effective_weight[:q_end]
                )
                blk.attn.k_proj.weight[:, -3:].copy_(
                    effective_weight[q_end:k_end]
                )
                blk.attn.v_proj.weight[:, -3:].copy_(
                    effective_weight[k_end:]
                )
>>>>>>> REPLACE
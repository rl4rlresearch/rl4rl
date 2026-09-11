MECHANISM: Noncontiguous fourth pre-attention LayerNorm scale absorption

HYPOTHESIS: A 1,377-parameter transformer will retain at least 99% accuracy because the qualified 1,378-parameter design reached 99.39%, while absorbing coordinate zero instead of extending the failed contiguous suffix tests whether the fourth-scale failure was coordinate-specific.

INTENDED_EDIT: Reproduce the qualified complete terminal gauge, four-column attention-output gauge, and three absorbed `ln1` scales, then absorb the additional `ln1` scale at coordinate zero with full ambient-gradient clipping and AdamW updates.

EVIDENCE: Reference Design 3 achieved 99.39% at 1,378 parameters using three-scale absorption; the contiguous fourth-scale trial fell to 49.67%, motivating a noncontiguous fourth coordinate while retaining the previously successful mechanism.

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
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
        self.weight_rest.copy_(raw_weight[:, 4:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
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
class NoncontiguousFourAbsorbedScaleLayerNorm(nn.Module):
    """LayerNorm with coordinate zero and its final three scales absorbed."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight.new_ones(1),
                self.weight,
                self.weight.new_ones(3),
            )
        )
        return F.layer_norm(
            x, (self.normalized_shape,), weight, None, self.eps
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = TwoAbsorbedScaleLayerNorm(cfg.d_model)
=======
        self.ln1 = NoncontiguousFourAbsorbedScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_rest.copy_(raw_weight[:, 3:])
                nn.init.zeros_(module.bias)
=======
                module.weight_rest.copy_(raw_weight[:, 4:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
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
        for blk in model.blocks
    ]
    attention_weight_v = [
        torch.zeros_like(moment) for moment in attention_weight_m
    ]
    attention_scale_m = [
        torch.zeros_like(scale) for scale in attention_scales
    ]
    attention_scale_v = [
        torch.zeros_like(moment) for moment in attention_scale_m
    ]
    attention_step = 0
=======
    # Coordinate zero and the final three ln1 scales live only in
    # optimizer state; q, k, and v store their effective products.
    attention_columns = [
        0,
        model_cfg.d_model - 3,
        model_cfg.d_model - 2,
        model_cfg.d_model - 1,
    ]
    attention_column_index = torch.tensor(
        attention_columns, device=device
    )
    attention_scales = [
        torch.ones(
            len(attention_columns),
            device=device,
            dtype=blk.attn.q_proj.weight.dtype,
        )
        for blk in model.blocks
    ]
    attention_weight_m = [
        torch.zeros_like(
            torch.cat(
                (
                    blk.attn.q_proj.weight.index_select(
                        1, attention_column_index
                    ),
                    blk.attn.k_proj.weight.index_select(
                        1, attention_column_index
                    ),
                    blk.attn.v_proj.weight.index_select(
                        1, attention_column_index
                    ),
                ),
                dim=0,
            )
        )
        for blk in model.blocks
    ]
    attention_weight_v = [
        torch.zeros_like(moment) for moment in attention_weight_m
    ]
    attention_scale_m = [
        torch.zeros_like(scale) for scale in attention_scales
    ]
    attention_scale_v = [
        torch.zeros_like(moment) for moment in attention_scale_m
    ]
    attention_step = 0
>>>>>>> REPLACE

<<<<<<< SEARCH
        attention_absorbed_grads = []
        for blk, virtual_scale in zip(
            model.blocks, attention_scales
        ):
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
        attention_absorbed_grads = []
        for blk, virtual_scale in zip(
            model.blocks, attention_scales
        ):
            effective_grad = torch.cat(
                (
                    blk.attn.q_proj.weight.grad.index_select(
                        1, attention_column_index
                    ),
                    blk.attn.k_proj.weight.grad.index_select(
                        1, attention_column_index
                    ),
                    blk.attn.v_proj.weight.grad.index_select(
                        1, attention_column_index
                    ),
                ),
                dim=0,
            ).detach().clone()
            effective_weight = torch.cat(
                (
                    blk.attn.q_proj.weight.index_select(
                        1, attention_column_index
                    ),
                    blk.attn.k_proj.weight.index_select(
                        1, attention_column_index
                    ),
                    blk.attn.v_proj.weight.index_select(
                        1, attention_column_index
                    ),
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
            blk.attn.q_proj.weight.grad.index_fill_(
                1, attention_column_index, 0.0
            )
            blk.attn.k_proj.weight.grad.index_fill_(
                1, attention_column_index, 0.0
            )
            blk.attn.v_proj.weight.grad.index_fill_(
                1, attention_column_index, 0.0
            )
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
                blk.attn.q_proj.weight.index_copy_(
                    1,
                    attention_column_index,
                    effective_weight[:q_end],
                )
                blk.attn.k_proj.weight.index_copy_(
                    1,
                    attention_column_index,
                    effective_weight[q_end:k_end],
                )
                blk.attn.v_proj.weight.index_copy_(
                    1,
                    attention_column_index,
                    effective_weight[k_end:],
                )
>>>>>>> REPLACE
MECHANISM: Second attention-output column gauge atop six terminal-column gauges

HYPOTHESIS: A 1,531-parameter model will retain at least 99% accuracy because the verified six-terminal-column design achieved 99.91%, while removing a second attention-projection column’s common-output component is an exact downstream-LayerNorm symmetry whose full AdamW dynamics are preserved in ambient coordinates.

INTENDED_EDIT: Extend the qualified terminal projection gauge from four to six columns and independently gauge-fix a second attention output-projection column, updating initialization, gradient clipping, and ambient AdamW bookkeeping accordingly.

EVIDENCE: Reference Design 3 achieved 99.91% accuracy with 1,532 parameters and six terminal-column gauges. The seventh terminal gauge failed at 68.48%, motivating a distinct reduction axis; the current qualified designs already demonstrate that the first attention-projection output gauge is compatible with high accuracy.

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and four weight-column output gauges removed."""

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
=======
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with bias and six weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(6)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 6)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_rest.copy_(raw_weight[:, 4:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
        self.weight_rest.copy_(raw_weight[:, 6:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with bias and one weight-column output gauge removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.Parameter(torch.empty(out_features - 1))
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 1)
        )
=======
class GaugeFixedAttentionProjection(nn.Module):
    """Linear projection with bias and two weight-column output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_prefix = nn.ParameterList(
            [
                nn.Parameter(torch.empty(out_features - 1))
                for _ in range(2)
            ]
        )
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 2)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight_prefix.copy_(
            raw_weight[:-1, 0] - raw_weight[-1, 0]
        )
        self.weight_rest.copy_(raw_weight[:, 1:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 2:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_weight_prefix = torch.cat(
            (self.weight_prefix, self.weight_prefix.new_zeros(1))
        )
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            full_weight_prefix.retain_grad()
            full_bias.retain_grad()
            self.full_weight_prefix = full_weight_prefix
            self.full_bias = full_bias
        weight = torch.cat(
            (full_weight_prefix.unsqueeze(1), self.weight_rest), dim=1
        )
        return F.linear(x, weight, full_bias)
=======
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
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_prefix.copy_(
                    raw_weight[:-1, 0] - raw_weight[-1, 0]
                )
                module.weight_rest.copy_(raw_weight[:, 1:])
                nn.init.zeros_(module.bias)
=======
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 2:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
                module.weight_rest.copy_(raw_weight[:, 4:])
                nn.init.zeros_(module.bias)
=======
                module.weight_rest.copy_(raw_weight[:, 6:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # positional, attention-weight, attention-bias, terminal-bias, and four
    # terminal-weight gauges.
    gauge_params = [model.token_emb.weight, model.pos_emb.first]
    for blk in model.blocks:
        gauge_params.append(blk.attn.proj.weight_prefix)
        gauge_params.append(blk.attn.proj.bias)
        gauge_params.append(blk.mlp.fc2.bias)
        gauge_params.extend(blk.mlp.fc2.weight_prefix)
=======
    # Preserve full ambient-coordinate AdamW dynamics for the tied-token,
    # positional, two attention-weight, attention-bias, terminal-bias, and six
    # terminal-weight gauges.
    gauge_params = [model.token_emb.weight, model.pos_emb.first]
    for blk in model.blocks:
        gauge_params.extend(blk.attn.proj.weight_prefix)
        gauge_params.append(blk.attn.proj.bias)
        gauge_params.append(blk.mlp.fc2.bias)
        gauge_params.extend(blk.mlp.fc2.weight_prefix)
>>>>>>> REPLACE

<<<<<<< SEARCH
            full_gauge_grads.append(
                blk.attn.proj.full_weight_prefix.grad.detach()
            )
            full_gauge_grads.append(
                blk.attn.proj.full_bias.grad.detach()
            )
=======
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.attn.proj.full_weight_prefix
            )
            full_gauge_grads.append(
                blk.attn.proj.full_bias.grad.detach()
            )
>>>>>>> REPLACE
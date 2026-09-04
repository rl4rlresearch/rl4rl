MECHANISM: Four-column terminal MLP output-direction gauge fixing

HYPOTHESIS: Extending the verified three-column terminal gauge to four `fc2` weight columns will reduce the model to 1,622 parameters while retaining at least 99% accuracy, because the additional all-ones output component is erased by the final LayerNorm and its full eight-coordinate AdamW dynamics are preserved.

INTENDED_EDIT: Gauge-fix the first four terminal MLP weight columns, reconstruct them during forward passes, and include every column in the existing ambient-coordinate optimizer.

EVIDENCE: The three-column design achieved 99.95% accuracy with 1,623 parameters, following successful one- and two-column designs; extending the same exact invariance by one adjacent column is the smallest evidence-supported reduction.

<<<<<<< SEARCH
class GaugeFixedTerminalLinear(nn.Module):
    """Linear layer with two all-ones output gauges removed."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.first_weight = nn.Parameter(torch.empty(out_features - 1))
        self.weight_rest = nn.Parameter(
            torch.empty(out_features, in_features - 1)
        )
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_first_weight = None
        self.full_bias = None
        self.reset_parameters()
=======
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
        self.bias = nn.Parameter(torch.empty(out_features - 1))
        self.full_weight_prefix = None
        self.full_bias = None
        self.reset_parameters()
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.first_weight.copy_(
            raw_weight[:-1, 0] - raw_weight[-1, 0]
        )
        self.weight_rest.copy_(raw_weight[:, 1:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
=======
        for column, stored in enumerate(self.weight_prefix):
            stored.copy_(
                raw_weight[:-1, column] - raw_weight[-1, column]
            )
        self.weight_rest.copy_(raw_weight[:, 4:])
        self.bias.copy_(raw_bias[:-1] - raw_bias[-1])
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_first_weight = torch.cat(
            (self.first_weight, self.first_weight.new_zeros(1))
        )
        full_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        if torch.is_grad_enabled():
            full_first_weight.retain_grad()
            full_bias.retain_grad()
            self.full_first_weight = full_first_weight
            self.full_bias = full_bias
        weight = torch.cat(
            (full_first_weight.unsqueeze(1), self.weight_rest), dim=1
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
                module.first_weight.copy_(
                    raw_weight[:-1, 0] - raw_weight[-1, 0]
                )
                module.weight_rest.copy_(raw_weight[:, 1:])
                nn.init.zeros_(module.bias)
=======
                for column, stored in enumerate(module.weight_prefix):
                    stored.copy_(
                        raw_weight[:-1, column] - raw_weight[-1, column]
                    )
                module.weight_rest.copy_(raw_weight[:, 4:])
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
    # Each stored seven-vector contains differences from an omitted eighth
    # coordinate. Preserve the full ambient AdamW dynamics for the positional
    # row, terminal bias, and one terminal weight column.
    gauge_params = [model.pos_emb.first]
    for blk in model.blocks:
        gauge_params.extend(
            [blk.mlp.fc2.bias, blk.mlp.fc2.first_weight]
        )
=======
    # Preserve full eight-coordinate AdamW dynamics for the positional,
    # terminal-bias, and four terminal-weight gauge vectors.
    gauge_params = [model.pos_emb.first]
    for blk in model.blocks:
        gauge_params.append(blk.mlp.fc2.bias)
        gauge_params.extend(blk.mlp.fc2.weight_prefix)
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
        for blk in model.blocks:
            full_gauge_grads.extend(
                [
                    blk.mlp.fc2.full_bias.grad.detach(),
                    blk.mlp.fc2.full_first_weight.grad.detach(),
                ]
            )
=======
        full_gauge_grads = [model.pos_emb.full_first.grad.detach()]
        for blk in model.blocks:
            full_gauge_grads.append(
                blk.mlp.fc2.full_bias.grad.detach()
            )
            full_gauge_grads.extend(
                full_weight.grad.detach()
                for full_weight in blk.mlp.fc2.full_weight_prefix
            )
>>>>>>> REPLACE
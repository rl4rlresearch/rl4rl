MECHANISM: Sixth attention-input LayerNorm scale gauge

HYPOTHESIS: Combining the verified 1,580-parameter QKV gauges with one additional anchored `ln1` scale will produce a 1,579-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reproduce query rows 0, 1, 4, and 5, key row 8, and value rows 16 and 17, then fix a sixth `ln1` scale to one; its effect remains learnable through the corresponding QKV input column.

EVIDENCE: The seven-row QKV design achieved 99.83% accuracy at 1,580 parameters, while further value-row gauges collapsed; an orthogonal LayerNorm-scale reparameterization is therefore the most informative one-parameter reduction.

<<<<<<< SEARCH
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model
        self.gauged_rows = (0,)
        self.ungauged_rows = tuple(range(1, self.out_features))

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(d_model, 3 * d_model)
        retained = self.out_features * d_model - len(self.gauged_rows)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.bias = nn.Parameter(base.bias.new_empty(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row_width = self.in_features - 1
        gauged = torch.cat(
            (
                self.weight[:row_width].view(1, row_width),
                self.weight.new_zeros(1, 1),
            ),
            dim=1,
        )
        ungauged = self.weight[row_width:].view(
            len(self.ungauged_rows), self.in_features
        )
        weight = torch.cat((gauged, ungauged), dim=0)

        fused_bias = torch.cat(
=======
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int, head_dim: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model
        self.gauged_rows = (
            0,
            1,
            head_dim,
            head_dim + 1,
            d_model,
            2 * d_model,
            2 * d_model + 1,
        )
        self.ungauged_rows = tuple(
            row for row in range(self.out_features) if row not in self.gauged_rows
        )

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(d_model, 3 * d_model)
        retained = self.out_features * d_model - len(self.gauged_rows)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.bias = nn.Parameter(base.bias.new_empty(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row_width = self.in_features - 1
        split = len(self.gauged_rows) * row_width
        gauged = torch.cat(
            (
                self.weight[:split].view(len(self.gauged_rows), row_width),
                self.weight.new_zeros(len(self.gauged_rows), 1),
            ),
            dim=1,
        )
        ungauged = self.weight[split:].view(
            len(self.ungauged_rows), self.in_features
        )

        rows = []
        gauged_index = 0
        ungauged_index = 0
        for row in range(self.out_features):
            if row in self.gauged_rows:
                rows.append(gauged[gauged_index])
                gauged_index += 1
            else:
                rows.append(ungauged[ungauged_index])
                ungauged_index += 1
        weight = torch.stack(rows)

        fused_bias = torch.cat(
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Retain query bias and one representative of QKV row 0's normalized-
        # input coefficient gauge.
        self.qkv = LayerNormGaugedQKV(d_model)
=======
        # Key and value biases are redundant; gauge selected normalized-input
        # rows while retaining the learned query bias.
        self.qkv = LayerNormGaugedQKV(d_model, self.head_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(5)))
=======
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(6)))
>>>>>>> REPLACE

<<<<<<< SEARCH
                # All attention scales initialize to one, so subtracting the
                # omitted coefficient preserves the initial row function.
=======
                # Attention scales initialize to one, so subtracting each
                # omitted coefficient preserves every selected row function.
>>>>>>> REPLACE
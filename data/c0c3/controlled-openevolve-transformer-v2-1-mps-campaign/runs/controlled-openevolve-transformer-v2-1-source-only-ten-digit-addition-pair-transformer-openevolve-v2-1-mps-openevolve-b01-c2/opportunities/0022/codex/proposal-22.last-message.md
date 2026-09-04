MECHANISM: Noncontiguous LayerNorm-nullspace weight gauge fixing

HYPOTHESIS: Gauge-fixing rows 0, 1, 2, and 4 of `fc1` will produce a 1,606-parameter model with at least 99% accuracy, testing whether the failed contiguous fourth-row reduction was specific to row 3 rather than a four-row capacity boundary.

INTENDED_EDIT: Generalize `LayerNormGaugedLinear` to omit the final coefficient from four selected rows, reconstruct those coefficients as zero, and gauge-adjust the same rows during initialization.

EVIDENCE: Gauging rows 0–2 achieved 99.97% at 1,607 parameters, while adding contiguous row 3 fell to 97.77%; substituting a different fourth row is the smallest direct test of row-specific optimization sensitivity.

<<<<<<< SEARCH
class LayerNormGaugedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(base.weight.new_empty(out_features * in_features - 2))
        self.bias = nn.Parameter(base.bias.new_empty(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row_width = self.in_features - 1
        weight = torch.cat(
            (
                self.weight[:row_width],
                self.weight.new_zeros(1),
                self.weight[row_width : 2 * row_width],
                self.weight.new_zeros(1),
                self.weight[2 * row_width :],
            )
        ).view(self.out_features, self.in_features)
        return F.linear(x, weight, self.bias)
=======
class LayerNormGaugedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gauged_rows = (0, 1, 2, 4)
        self.ungauged_rows = tuple(
            row for row in range(out_features) if row not in self.gauged_rows
        )

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(in_features, out_features)
        retained = out_features * in_features - len(self.gauged_rows)
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.bias = nn.Parameter(base.bias.new_empty(out_features))

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
        return F.linear(x, weight, self.bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
                # For normalized z, sum(z) is zero. Each output row therefore
                # has an independent common-coefficient null direction.
                full[:2, :-1].sub_(full[:2, -1:].clone())
                module.weight.copy_(
                    torch.cat((full[:2, :-1].flatten(), full[2:].flatten()))
                )
                nn.init.zeros_(module.bias)
=======
                # For normalized z, sum(z) is zero. Each selected output row
                # therefore has an independent common-coefficient null direction.
                gauged = full[list(module.gauged_rows)].clone()
                gauged[:, :-1].sub_(gauged[:, -1:])
                ungauged = full[list(module.ungauged_rows)]
                module.weight.copy_(
                    torch.cat((gauged[:, :-1].flatten(), ungauged.flatten()))
                )
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE
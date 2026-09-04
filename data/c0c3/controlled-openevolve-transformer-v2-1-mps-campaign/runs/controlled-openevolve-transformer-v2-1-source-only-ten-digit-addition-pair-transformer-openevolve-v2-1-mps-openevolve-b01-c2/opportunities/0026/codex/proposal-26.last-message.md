MECHANISM: Eighth noncontiguous LayerNorm-nullspace weight gauge

HYPOTHESIS: Gauge-fixing `fc1` rows 0, 1, 2, 4, 5, 6, 7, and 8 will produce a 1,602-parameter model with at least 99% accuracy.

INTENDED_EDIT: Extend the verified noncontiguous gauge set with row 8, omitting its final input coefficient and reconstructing it as zero while preserving gauge-equivalent initialization.

EVIDENCE: Gauging rows 0, 1, 2, 4, 5, 6, and 7 achieved 99.94% accuracy at 1,603 parameters; adding row 8 is the smallest incremental reduction that continues to avoid sensitive row 3.

<<<<<<< SEARCH
class LayerNormGaugedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gauged_rows = 3

        # Consume the same constructor RNG stream as the replaced nn.Linear.
        base = nn.Linear(in_features, out_features)
        retained = out_features * in_features - self.gauged_rows
        self.weight = nn.Parameter(base.weight.new_empty(retained))
        self.bias = nn.Parameter(base.bias.new_empty(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        row_width = self.in_features - 1
        split = self.gauged_rows * row_width
        gauged = torch.cat(
            (
                self.weight[:split].view(self.gauged_rows, row_width),
                self.weight.new_zeros(self.gauged_rows, 1),
            ),
            dim=1,
        )
        tail = self.weight[split:].view(
            self.out_features - self.gauged_rows, self.in_features
        )
        weight = torch.cat((gauged, tail), dim=0)
        return F.linear(x, weight, self.bias)
=======
class LayerNormGaugedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gauged_rows = (0, 1, 2, 4, 5, 6, 7, 8)
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
        if isinstance(module, LayerNormGaugedLinear):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                rows = module.gauged_rows
                full[:rows, :-1].sub_(full[:rows, -1:].clone())
                module.weight.copy_(
                    torch.cat(
                        (full[:rows, :-1].flatten(), full[rows:].flatten())
                    )
                )
                nn.init.zeros_(module.bias)
=======
        if isinstance(module, LayerNormGaugedLinear):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                # LayerNorm outputs sum to zero, so each selected output row
                # has an independent common-coefficient null direction.
                gauged = full[list(module.gauged_rows)].clone()
                gauged[:, :-1].sub_(gauged[:, -1:])
                ungauged = full[list(module.ungauged_rows)]
                module.weight.copy_(
                    torch.cat((gauged[:, :-1].flatten(), ungauged.flatten()))
                )
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE
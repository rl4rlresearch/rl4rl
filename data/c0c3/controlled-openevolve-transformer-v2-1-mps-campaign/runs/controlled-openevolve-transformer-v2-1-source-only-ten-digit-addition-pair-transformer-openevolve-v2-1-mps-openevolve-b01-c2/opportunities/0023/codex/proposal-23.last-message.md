MECHANISM: Fifth noncontiguous LayerNorm-nullspace weight gauge

HYPOTHESIS: Gauge-fixing `fc1` rows 0, 1, 2, 4, and 5 will produce a 1,605-parameter model with at least 99% accuracy, because rows 0, 1, 2, and 4 already achieved 99.95% while the failed contiguous experiment specifically implicated row 3.

INTENDED_EDIT: Anchor one scale coordinate in both pre-sublayer LayerNorms and replace `fc1` with a linear layer that omits the final coefficient from five selected rows, reconstructing those coefficients as zero and gauge-adjusting initialization.

EVIDENCE: The noncontiguous four-row design reached 99.95% at 1,606 parameters, whereas adding row 3 to rows 0–2 reached only 97.77%; extending the successful set with row 5 is the smallest probe that avoids the identified sensitive row.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
=======
class LayerNormGaugedLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.gauged_rows = (0, 1, 2, 4, 5)
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


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = LayerNormGaugedLinear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln1 = AnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, LayerNormGaugedLinear):
            with torch.no_grad():
                full = module.weight.new_empty(module.out_features, module.in_features)
                nn.init.normal_(full, mean=0.0, std=0.02)

                # For normalized z, sum(z) is zero, giving each selected row
                # an independent common-coefficient null direction.
                gauged = full[list(module.gauged_rows)].clone()
                gauged[:, :-1].sub_(gauged[:, -1:])
                ungauged = full[list(module.ungauged_rows)]
                module.weight.copy_(
                    torch.cat((gauged[:, :-1].flatten(), ungauged.flatten()))
                )
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE
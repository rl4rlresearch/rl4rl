MECHANISM: Fourth-row LayerNorm-nullspace weight gauge fixing

HYPOTHESIS: Gauge-fixing the final input-weight coordinate in the first four `fc1` rows will produce a 1,606-parameter model with at least 99% accuracy, because the verified three-row implementation achieved 99.97% and each additional row has the same normalized-input null direction.

INTENDED_EDIT: Anchor one scale coordinate in both pre-sublayer LayerNorms and replace `fc1` with a linear layer that omits one coefficient from each of its first four rows while preserving equivalent initialization.

EVIDENCE: The 1,607-parameter three-row gauge-fixed design achieved 99.97% accuracy; extending the same row-local reduction by one row is the smallest directly supported next probe.

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
        self.gauged_rows = 4

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
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE
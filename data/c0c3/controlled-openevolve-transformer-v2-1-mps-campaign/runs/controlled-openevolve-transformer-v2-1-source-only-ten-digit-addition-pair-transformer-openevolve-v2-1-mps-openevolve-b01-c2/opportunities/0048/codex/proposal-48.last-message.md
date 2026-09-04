MECHANISM: Second-head query-row LayerNorm-nullspace gauge

HYPOTHESIS: Reproducing the verified four-row QKV gauge and additionally gauge-fixing query row 6 will produce a 1,582-parameter model with at least 99% accuracy.

INTENDED_EDIT: Anchor five `ln1` scales and omit the final normalized-input coefficient from query rows 0, 1, 4, 5, and 6 while preserving their initialized functions.

EVIDENCE: The balanced rows 0, 1, 4, and 5 design achieved 99.98% accuracy at 1,583 parameters; adding one row in the second head is the smallest reduction from that high-margin result and avoids the previously unsuccessful row-2 placement.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int, head_dim: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model
        self.gauged_rows = (0, 1, head_dim, head_dim + 1, head_dim + 2)
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
            (
                self.bias,
                self.bias.new_zeros(self.in_features),
                self.bias.new_zeros(self.in_features),
            )
        )
        return F.linear(x, weight, fused_bias)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Key bias cancels in softmax, and value bias is absorbed by proj.bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
=======
        # Retain query bias while gauge-fixing selected normalized-input rows.
        self.qkv = LayerNormGaugedQKV(d_model, self.head_dim)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        q_bias = self.qkv.bias
        fused_bias = torch.cat(
            (q_bias, q_bias.new_zeros(d_model), q_bias.new_zeros(d_model))
        )
        qkv = F.linear(x, self.qkv.weight, fused_bias)
        q, k, v = qkv.chunk(3, dim=-1)
=======
        bsz, seqlen, d_model = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(4)))
=======
        self.weight = nn.Parameter(torch.ones(d_model - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(5)))
>>>>>>> REPLACE

<<<<<<< SEARCH
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, LayerNormGaugedLinear):
=======
    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, LayerNormGaugedQKV):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                # Scales initialize to one, so subtracting each omitted
                # coefficient preserves every selected row function.
                gauged = full[list(module.gauged_rows)].clone()
                gauged[:, :-1].sub_(gauged[:, -1:])
                ungauged = full[list(module.ungauged_rows)]
                module.weight.copy_(
                    torch.cat((gauged[:, :-1].flatten(), ungauged.flatten()))
                )
                nn.init.zeros_(module.bias)
        elif isinstance(module, LayerNormGaugedLinear):
>>>>>>> REPLACE
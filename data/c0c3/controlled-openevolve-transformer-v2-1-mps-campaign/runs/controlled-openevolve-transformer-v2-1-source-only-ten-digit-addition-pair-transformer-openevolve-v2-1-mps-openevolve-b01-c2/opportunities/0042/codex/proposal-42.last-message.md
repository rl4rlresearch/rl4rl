MECHANISM: Second-row QKV normalized-input gauge

HYPOTHESIS: Gauge-fixing QKV row 1 alongside the verified row 0 gauge will reduce the model from 1,586 to 1,585 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Replace the fused QKV linear layer with a learned projection that omits the final input coefficient from query rows 0 and 1, reconstructs those coefficients as zero, retains query bias, and preserves the initialized row functions.

EVIDENCE: The single-row QKV gauge achieved 99.94% accuracy at 1,586 parameters. The prior two-row attempt could not be verified and supplied no accuracy result, so a correctly generalized second-row gauge is the smallest informative continuation.

<<<<<<< SEARCH
class CausalSelfAttention(nn.Module):
=======
class LayerNormGaugedQKV(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.in_features = d_model
        self.out_features = 3 * d_model
        self.gauged_rows = (0, 1)
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
        # Preserve fused projection construction while retaining only query bias.
        # Key bias cancels in softmax and value bias is absorbed by proj.bias.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model))
=======
        # Retain query bias while gauge-fixing two normalized-input QKV rows.
        self.qkv = LayerNormGaugedQKV(d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = self.qkv.bias
        fused_bias = torch.cat(
            (q_bias, q_bias.new_zeros(d_model), q_bias.new_zeros(d_model))
        )
        qkv = F.linear(x, self.qkv.weight, fused_bias)
=======
        qkv = self.qkv(x)
>>>>>>> REPLACE

<<<<<<< SEARCH
        if isinstance(module, LayerNormGaugedLinear):
=======
        if isinstance(module, LayerNormGaugedQKV):
            with torch.no_grad():
                full = module.weight.new_empty(
                    module.out_features, module.in_features
                )
                nn.init.normal_(full, mean=0.0, std=0.02)

                # Attention scales initialize to one, so subtracting each
                # omitted coefficient preserves both selected row functions.
                gauged = full[list(module.gauged_rows)].clone()
                gauged[:, :-1].sub_(gauged[:, -1:])
                ungauged = full[list(module.ungauged_rows)]
                module.weight.copy_(
                    torch.cat((gauged[:, :-1].flatten(), ungauged.flatten()))
                )
                nn.init.zeros_(module.bias)
        elif isinstance(module, LayerNormGaugedLinear):
>>>>>>> REPLACE
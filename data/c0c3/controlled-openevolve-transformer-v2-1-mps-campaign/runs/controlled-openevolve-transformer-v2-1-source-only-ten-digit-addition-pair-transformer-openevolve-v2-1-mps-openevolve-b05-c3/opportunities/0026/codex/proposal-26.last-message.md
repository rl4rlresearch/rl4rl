MECHANISM: LayerNorm-hyperplane gauge for one MLP input weight

HYPOTHESIS: Quotienting one `fc1` weight along the exact zero-mean LayerNorm input direction, while retaining unit 10’s learned threshold, will reduce the qualified design to 1599 parameters and maintain at least 99% accuracy.

INTENDED_EDIT: Use four learned query-bias coordinates, retain the qualified fixed-zero trailing `fc1` bias, and omit unit 10’s final input weight after gauge-equivalent initialization from the full baseline draw.

EVIDENCE: The 1600-parameter design with one fixed `fc1` bias achieved 99.91%, whereas directly removing a second bias reached 98.57% and removing an `fc2` connection reached 55.04%; this motivates removing an exact LayerNorm-mediated affine redundancy while preserving every hidden unit’s usable threshold and output connection.

<<<<<<< SEARCH
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Five query coordinates are
        # learned; the remaining query coordinates and key/value biases are zero.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 3))
=======
        # Construct with the baseline shape first so subsequent modules retain
        # the proven initialization RNG stream. Four query coordinates are
        # learned; the remaining query coordinates and key/value biases are zero.
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = nn.Parameter(torch.empty(d_model - 4))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 3))
        )
=======
        bias = torch.cat(
            (self.qkv.bias, self.qkv.bias.new_zeros(2 * d_model + 4))
        )
>>>>>>> REPLACE

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
class GaugeFixedFC1(nn.Linear):
    """MLP input map with one LayerNorm-invisible weight coordinate omitted."""

    def __init__(self, in_features: int, out_features: int):
        # Consume the constructor RNG of the original full linear layer.
        super().__init__(in_features, out_features)
        self.weight = nn.Parameter(
            torch.empty(out_features * in_features - 1)
        )
        self.bias = nn.Parameter(torch.empty(out_features - 1))

    @property
    def omitted_index(self) -> int:
        # Omit the final input coordinate of the penultimate hidden unit; its
        # learned bias absorbs the affine-hyperplane intercept.
        return (self.out_features - 1) * self.in_features - 1

    def full_weight(self) -> torch.Tensor:
        zero = self.weight.new_zeros(1)
        flat = torch.cat(
            (
                self.weight[: self.omitted_index],
                zero,
                self.weight[self.omitted_index :],
            )
        )
        return flat.view(self.out_features, self.in_features)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = GaugeFixedFC1(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.fc1.bias, self.fc1.bias.new_zeros(1)))
        hidden = F.linear(x, self.fc1.full_weight(), bias)
        return self.drop(self.fc2(F.gelu(hidden)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
=======
        elif isinstance(module, GaugeFixedFC1):
            # Draw the original full matrix to preserve the initialization
            # stream. LayerNorm initially has unit scale and zero offset, so
            # subtracting the omitted coefficient from its entire row leaves
            # the hidden preactivation unchanged on the zero-mean input plane.
            full = module.weight.new_empty(
                module.out_features, module.in_features
            )
            nn.init.normal_(full, mean=0.0, std=0.02)
            row = module.out_features - 2
            with torch.no_grad():
                full[row].sub_(full[row, -1])
                flat = full.reshape(-1)
                module.weight.copy_(
                    torch.cat(
                        (
                            flat[: module.omitted_index],
                            flat[module.omitted_index + 1 :],
                        )
                    )
                )
                nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
>>>>>>> REPLACE
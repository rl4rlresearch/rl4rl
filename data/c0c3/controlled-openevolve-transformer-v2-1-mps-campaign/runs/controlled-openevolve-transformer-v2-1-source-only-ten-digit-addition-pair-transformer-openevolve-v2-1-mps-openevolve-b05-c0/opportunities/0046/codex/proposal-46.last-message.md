MECHANISM: Coordinate-4 post-MLP residual-shift quotient

HYPOTHESIS: Fixing MLP output-bias index 4 will reduce the model from 1622 to 1621 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Replace the MLP output projection with an initialization-preserving projection whose eight-dimensional bias is reconstructed with coordinate 4 fixed to zero, and optimize its seven learned coordinates through the existing virtual-coordinate AdamW path.

EVIDENCE: The previously tested MLP coordinates 0, 1, and 7 failed, demonstrating coordinate sensitivity; value-bias coordinate 4 is the strongest untested analogue because its later removal achieved 100% accuracy despite an earlier adverse result.

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
class GaugeFixedMLPOutput(nn.Module):
    """MLP output projection with bias coordinate 4 fixed to zero."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fixed_index = 4

        # Match nn.Linear's constructor-time random-number consumption.
        source = nn.Linear(in_features, out_features)
        self.weight = nn.Parameter(source.weight.detach().clone())
        full_bias = source.bias.detach()
        self.bias = nn.Parameter(
            torch.cat(
                (
                    full_bias[: self.fixed_index],
                    full_bias[self.fixed_index + 1 :],
                )
            ).clone()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        full_bias = torch.cat(
            (
                self.bias[: self.fixed_index],
                self.bias.new_zeros(1),
                self.bias[self.fixed_index :],
            )
        )
        return F.linear(x, self.weight, full_bias)


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = GaugeFixedMLPOutput(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
>>>>>>> REPLACE

<<<<<<< SEARCH
        elif isinstance(module, GaugeFixedQKV):
            full_weight = torch.empty(
                3 * module.d_model,
                module.d_model,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full_(full_weight)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
=======
        elif isinstance(module, GaugeFixedQKV):
            full_weight = torch.empty(
                3 * module.d_model,
                module.d_model,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            nn.init.normal_(full_weight, mean=0.0, std=0.02)
            module.reset_from_full_(full_weight)
        elif isinstance(module, GaugeFixedMLPOutput):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            nn.init.zeros_(module.bias)
        elif isinstance(module, (nn.Linear, nn.Embedding)):
>>>>>>> REPLACE

<<<<<<< SEARCH
    gauge_parameters = [
        model.token_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
    ]
=======
    gauge_parameters = [
        model.token_emb.weight,
        *[block.attn.proj_bias for block in model.blocks],
        *[block.mlp.fc2.bias for block in model.blocks],
    ]
>>>>>>> REPLACE
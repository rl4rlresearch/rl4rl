MECHANISM: Coordinate-specific post-MLP residual-shift quotient

HYPOTHESIS: Fixing MLP output-bias index 1 instead of the previously failed final coordinate will reduce the model from 1622 to 1621 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Store the seven MLP output-bias coordinates other than index 1, reconstruct index 1 as zero, and train the reduced bias with the existing virtual-coordinate AdamW and gradient-clipping logic.

EVIDENCE: The final-coordinate MLP-bias gauge failed at 7.8%, but value-bias reductions showed extreme coordinate sensitivity: index 1 succeeded at 99.98% while several other coordinates failed. This directly motivates testing index 1 for the same exact residual-shift quotient.

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
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2.register_parameter("bias", None)
        self.fc2_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = torch.cat(
            (
                self.fc2_bias[:1],
                self.fc2_bias.new_zeros(1),
                self.fc2_bias[1:],
            )
        )
        return self.drop(F.linear(hidden, self.fc2.weight, bias))
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
        *[block.mlp.fc2_bias for block in model.blocks],
    ]
>>>>>>> REPLACE
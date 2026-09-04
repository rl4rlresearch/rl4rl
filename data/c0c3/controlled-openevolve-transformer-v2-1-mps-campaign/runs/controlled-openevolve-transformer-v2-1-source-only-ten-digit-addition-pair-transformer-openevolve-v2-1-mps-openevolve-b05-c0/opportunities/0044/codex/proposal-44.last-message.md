MECHANISM: Coordinate-specific post-MLP residual-shift quotient

HYPOTHESIS: Fixing MLP output-bias index 0 will reduce the verified model from 1622 to 1621 parameters while maintaining at least 99% accuracy; value-bias evidence identifies index 0 as substantially more trainable than the previously failed MLP coordinates 1 and 7.

INTENDED_EDIT: Store MLP output-bias coordinates 1–7, reconstruct coordinate 0 as zero, and include the reduced bias in the existing virtual-coordinate AdamW and gradient-clipping path.

EVIDENCE: MLP output-bias removals at indices 7 and 1 reached only 7.8% and 49.89%, but analogous value-bias gauges were strongly coordinate-sensitive: index 0 reached 96.92% in its earlier adverse configuration and later supported 99.94%, motivating a coordinate-specific test rather than abandoning this exact quotient.

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
        self.fc2.bias = nn.Parameter(torch.zeros(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        bias = F.pad(self.fc2.bias, (1, 0))
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
        *[block.mlp.fc2.bias for block in model.blocks],
    ]
>>>>>>> REPLACE
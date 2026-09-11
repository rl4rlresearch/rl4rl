MECHANISM: Pre-nonlinearity LayerNorm/MLP bias sharing

HYPOTHESIS: Tying `ln2.bias` to the first eight coordinates of `fc1.bias` will reduce the model from 1628 to 1620 parameters while retaining at least 99% accuracy, because `fc1` can absorb the LayerNorm shift and the shared parameter preserves both optimization pathways.

INTENDED_EDIT: Disable the standalone second LayerNorm bias and reuse the first `d_model` entries of the MLP input bias as its affine shift before `fc1`.

EVIDENCE: Removing `ln2.bias` scored only 14.87%, while tying it to the post-nonlinearity `fc2.bias` scored 11.24%; sharing it with the mathematically corresponding pre-nonlinearity `fc1.bias` tests the same redundancy without deleting its gradient pathway or coupling it across GELU.

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
        if d_ff < d_model:
            raise ValueError("d_ff must be at least d_model for shared LayerNorm bias")
        self.d_model = d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.fc1.bias[:self.d_model]
        return self.drop(self.fc2(F.gelu(self.fc1(x))))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
=======
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
        self.mlp = MLP(cfg.d_model, cfg.d_ff, cfg.dropout)
>>>>>>> REPLACE
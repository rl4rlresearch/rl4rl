MECHANISM: Learned cross-sublayer residual-bias gauge tie

HYPOTHESIS: Tying coordinate 1 of the MLP output bias to coordinate 1 of the attention projection bias will reduce the model from 1,614 to 1,613 parameters while retaining at least 99% accuracy, because uniform opposite shifts of these biases leave the block function unchanged and a learned tie avoids the failed fixed-zero gauge constraints.

INTENDED_EDIT: Store seven `fc2` bias values, reconstruct its missing coordinate from the learned attention projection bias, and pass that bias into the MLP forward path.

EVIDENCE: The 1,614-parameter design achieved 99.91%, and learned coordinate-1 tying succeeded where fixed-zero key constraints failed; prior projection-bias gauge reductions used fixed or centered constraints and failed, motivating a learned cross-sublayer tie that retains the complete function class.

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
        self.fc2.bias = nn.Parameter(torch.empty(d_model - 1))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, attn_bias: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x))
        fc2_bias = torch.cat(
            (self.fc2.bias[:1], attn_bias[1:2], self.fc2.bias[1:])
        )
        return self.drop(F.linear(hidden, self.fc2.weight, fc2_bias))
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x), self.attn.proj.bias)
        return x
>>>>>>> REPLACE
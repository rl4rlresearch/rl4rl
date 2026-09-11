MECHANISM: Metric-tied analysis–synthesis MLP

HYPOTHESIS: Coupling the MLP’s analysis and synthesis dictionaries through learned feature gains and an 8×8 channel metric will retain at least 99% accuracy after 45,000 updates while reducing the model from 1,138 to 1,118 learned parameters.

INTENDED_EDIT: Replace the independent 8×12 MLP output matrix with the transpose of the learned 12×8 input dictionary, twelve learned feature gains, and a learned 8×8 channel mixer initialized to identity.

EVIDENCE: The 1,138-parameter backbone reached 99.78%, while further attention-output pruning, cross-head tying, embedding pruning, and output-bias pruning failed sharply. This leaves intact those sensitive paths and instead challenges the shared assumption that each nonlinear MLP feature requires unrelated input and output directions.

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
        # Analyze and synthesize with one learned feature dictionary. Per-feature
        # gains and a dense residual-space metric retain asymmetric amplitudes
        # and channel mixing using 20 fewer parameters than two independent
        # dense matrices. Constructing the original shape preserves RNG use.
        self.fc2 = nn.Linear(d_ff, d_model)
        self.fc2.weight = nn.Parameter(torch.empty(d_model, d_model))
        self.fc2.in_features = d_model
        self.fc2.analysis_synthesis = True
        self.feature_gain = nn.Parameter(torch.ones(d_ff))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = F.gelu(self.fc1(x)) * self.feature_gain
        decoded = F.linear(hidden, self.fc1.weight.t())
        return self.drop(self.fc2(decoded))
>>>>>>> REPLACE

<<<<<<< SEARCH
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
=======
            elif hasattr(module, "analysis_synthesis"):
                nn.init.eye_(module.weight)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
>>>>>>> REPLACE
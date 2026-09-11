MECHANISM: Cross-branch bias-coordinate sharing

HYPOTHESIS: Sharing one MLP output-bias coordinate with the learned attention query offset will reduce the model from 1,373 to 1,372 parameters while retaining at least 99% accuracy within 50,000 steps, because both successful 1,373- and 1,376-parameter designs already use bias coordinates for multiple roles without changing their zero initialization.

INTENDED_EDIT: Replace the seven-parameter `fc2` bias with six dedicated coordinates plus the attention projection-bias mean as the seventh coordinate, while preserving the same seven-coordinate MLP output, derived hidden bias, and optimized `F.linear` computation.

EVIDENCE: The 1,373-parameter model achieved 100% accuracy after reusing the MLP output-bias mean as its hidden bias, and the 1,376-parameter model achieved 100% after reusing the attention output-bias mean as its query offset; sharing one coordinate across these proven learned-bias mechanisms is the smallest direct next reduction.

<<<<<<< SEARCH
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = MeanZeroInputLinear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model - 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.fc1(x) + self.fc2.bias.mean()
        output = self.fc2(F.gelu(hidden))
        output = F.pad(output, (0, 1))
        return self.drop(output)
=======
class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.fc1 = MeanZeroInputLinear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model - 1, bias=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 2))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, shared_bias: torch.Tensor) -> torch.Tensor:
        output_bias = torch.cat((self.output_bias, shared_bias.reshape(1)))
        hidden = self.fc1(x) + output_bias.mean()
        output = F.linear(F.gelu(hidden), self.fc2.weight, output_bias)
        output = F.pad(output, (0, 1))
        return self.drop(output)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_bias = self.attn.proj.bias.mean()
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x), shared_bias)
        return x
>>>>>>> REPLACE
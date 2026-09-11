MECHANISM: Single-neuron LayerNorm-nullspace elimination

HYPOTHESIS: A 1,403-parameter model will retain at least 99% accuracy because it preserves the verified three-quartet bias design and removes only one functionally null MLP weight direction, leaving the other eleven neuron projections conventionally parameterized.

INTENDED_EDIT: Share MLP biases across three learned quartets and parameterize one MLP input-weight row in the seven-dimensional mean-zero subspace produced by its preceding LayerNorm.

EVIDENCE: The 1,404-parameter three-quartet design achieved 99.57%. Eliminating all 36 LayerNorm-null directions at once fell to 84.82%, motivating a one-row ablation that retains the same effective function class while minimizing the optimization disturbance.

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer with one learned-bias quartet and learned-bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 4 or out_features % 2 != 0:
            raise ValueError("out_features must be even and at least four")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 2 - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quartet_bias = self.free_bias[:1].repeat_interleave(4)
        paired_biases = self.free_bias[1:].repeat_interleave(2)
        bias = torch.cat((quartet_bias, paired_biases))
        return F.linear(x, self.weight, bias)
=======
class PairwiseTiedBiasLinear(nn.Module):
    """MLP projection with quartet biases and one mean-zero input row."""

    def __init__(self, in_features: int, out_features: int):
        if out_features % 4 != 0:
            raise ValueError("out_features must be divisible by four")
        super().__init__()
        self.full_rows = nn.Parameter(torch.empty(out_features - 1, in_features))
        self.reduced_row = nn.Parameter(torch.empty(in_features - 1))
        self.free_bias = nn.Parameter(torch.zeros(out_features // 4))
        self.register_buffer(
            "input_basis", mean_zero_basis(in_features), persistent=False
        )
        nn.init.normal_(self.full_rows, mean=0.0, std=0.02)
        nn.init.normal_(self.reduced_row, mean=0.0, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        first_row = self.reduced_row @ self.input_basis.transpose(0, 1)
        weight = torch.cat((first_row.unsqueeze(0), self.full_rows), dim=0)
        bias = self.free_bias.repeat_interleave(4)
        return F.linear(x, weight, bias)
>>>>>>> REPLACE
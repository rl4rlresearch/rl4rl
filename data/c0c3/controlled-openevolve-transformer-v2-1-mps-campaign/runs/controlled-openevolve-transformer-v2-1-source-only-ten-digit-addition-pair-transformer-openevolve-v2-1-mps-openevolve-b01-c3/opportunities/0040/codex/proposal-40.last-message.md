MECHANISM: Shared query-key metric attention

HYPOTHESIS: A 1,340-parameter model will retain at least 99% accuracy because each attention head can route through a full-width learned similarity metric, while independent value projections and the fixed query bias preserve content capacity and directional asymmetry.

INTENDED_EDIT: Start from the verified three-quartet MLP-bias design, then replace independent query and key projections with one shared learned projection while retaining separate full-width values.

EVIDENCE: The 1,404-parameter three-quartet design achieved 99.57%. Shared values failed at 60.09%, and reduced query/key width reportedly hurt, so this patch preserves independent values, both heads, and four routing dimensions per head while challenging only the untested assumption that query and key require separate coordinate maps.

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
=======
        self.qk = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
>>>>>>> REPLACE

<<<<<<< SEARCH
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias = F.pad(
=======
        q = self.qk(x)
        k = self.qk(x)
        v = self.v(x)
        q_bias = F.pad(
>>>>>>> REPLACE

<<<<<<< SEARCH
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer whose final six outputs form three learned-bias pairs."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_biases = self.free_bias[-3:].repeat_interleave(2)
        bias = torch.cat((self.free_bias[:-3], shared_biases))
        return F.linear(x, self.weight, bias)
=======
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer whose outputs form learned-bias quartets."""

    def __init__(self, in_features: int, out_features: int):
        if out_features % 4 != 0:
            raise ValueError("out_features must be divisible by four")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.free_bias.repeat_interleave(4)
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE
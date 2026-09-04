MECHANISM: Adjacent hidden-quartet bias sharing

HYPOTHESIS: A 1,271-parameter model will achieve at least 99% accuracy because it preserves the twice-verified fourteen-bin positional design while tying only two already quartet-shared MLP hidden biases.

INTENDED_EDIT: Extend far-distance sharing from twelve to fourteen bins and tie the last two MLP hidden-bias quartets, reducing the verified 1,272-parameter design by one parameter.

EVIDENCE: The fourteen-bin design achieved 99.30% and 99.28%; prior 1,271-parameter failures altered load-bearing positional biases, the MLP output, or QKV parameterization, whereas this tests an untried reduction in the already aggressively shared hidden biases.

<<<<<<< SEARCH
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 13))
        self.far_rel_bias = nn.Parameter(torch.zeros(12))
=======
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 15))
        self.far_rel_bias = nn.Parameter(torch.zeros(14))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.far_rel_bias.expand(self.n_head, 12),
=======
                self.far_rel_bias.expand(self.n_head, 14),
>>>>>>> REPLACE

<<<<<<< SEARCH
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
=======
class PairwiseTiedBiasLinear(nn.Linear):
    """Linear layer with bias quartets, sharing the final two quartets."""

    def __init__(self, in_features: int, out_features: int):
        if out_features < 8 or out_features % 4 != 0:
            raise ValueError("out_features must be a multiple of four and at least eight")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(out_features // 4 - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        quartet_bias = torch.cat((self.free_bias, self.free_bias[-1:]))
        bias = quartet_bias.repeat_interleave(4)
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE
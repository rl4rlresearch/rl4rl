MECHANISM: Global MLP hidden-bias sharing

HYPOTHESIS: A 1,270-parameter model will achieve at least 99% accuracy because it preserves the successful fourteen-bin positional design while replacing the 1,271-parameter model’s two hidden-bias values with one shared value.

INTENDED_EDIT: Extend far-distance sharing from eleven to fourteen bins and share one learned bias across all three MLP hidden quartets, reducing the current model by five parameters and the best verified model by one.

EVIDENCE: The 1,271-parameter reference achieved 99.04% after tying its final two hidden-bias quartets; extending that same reduction by one parameter tests the remaining quartet distinction without altering the twice-verified attention design.

<<<<<<< SEARCH
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 12))
        self.far_rel_bias = nn.Parameter(torch.zeros(11))
=======
        self.rel_bias = nn.Parameter(torch.zeros(n_head, max_seq_len - 15))
        self.far_rel_bias = nn.Parameter(torch.zeros(14))
>>>>>>> REPLACE

<<<<<<< SEARCH
                self.far_rel_bias.expand(self.n_head, 11),
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
    """Linear layer whose outputs share one learned hidden bias."""

    def __init__(self, in_features: int, out_features: int):
        if out_features % 4 != 0:
            raise ValueError("out_features must be divisible by four")
        super().__init__(in_features, out_features, bias=True)
        self.bias = None
        self.free_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.free_bias.expand(self.out_features)
        return F.linear(x, self.weight, bias)
>>>>>>> REPLACE
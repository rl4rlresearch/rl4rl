MECHANISM: Low-side nonadjacent attention-output bias anchoring

HYPOTHESIS: Extending the qualified 1503-parameter design by fixing reduced attention-output bias coordinate 1 will yield 1502 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Reproduce the qualified `q_bias[6]` anchor and fix reduced projection-bias coordinates 0, 1, 5, and 6, leaving coordinates 2–4 learned.

EVIDENCE: Anchoring coordinates 0, 5, and 6 achieved 99.93% accuracy at 1503 parameters, while anchoring coordinate 4 with 5 and 6 failed at 39.49%; coordinate 1 is the closest untested extension on the successful low-coordinate side.

<<<<<<< SEARCH
class AnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with one bias coordinate absorbed by value bias."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (0, 1))
        return F.linear(x, weight, bias)
=======
class LowSplitFourAnchoredMeanZeroOutputLinear(MeanZeroOutputLinear):
    """Mean-zero output map with four split bias coordinates fixed at zero."""
    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.bias = nn.Parameter(self.bias.detach()[2:-2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = self.basis @ self.weight
        bias = self.basis @ F.pad(self.bias, (2, 2))
        return F.linear(x, weight, bias)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 4))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 5))
        self.proj = LowSplitFourAnchoredMeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat(
            (
                self.q_bias.new_zeros(3),
                self.q_bias[:self.head_dim - 3],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim - 3:],
            )
        )
=======
        q_bias = torch.cat(
            (
                self.q_bias.new_zeros(3),
                self.q_bias[:self.head_dim - 3],
                self.q_bias.new_zeros(1),
                self.q_bias[
                    self.head_dim - 3:self.head_dim - 2
                ],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim - 2:],
            )
        )
>>>>>>> REPLACE
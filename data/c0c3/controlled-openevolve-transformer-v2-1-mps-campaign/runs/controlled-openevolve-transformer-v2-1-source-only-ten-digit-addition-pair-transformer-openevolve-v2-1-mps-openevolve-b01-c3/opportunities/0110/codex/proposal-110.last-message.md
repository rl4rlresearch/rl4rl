MECHANISM: Scalar-mixed shared value codebook

HYPOTHESIS: A 1,056-parameter transformer will achieve at least 99% accuracy because positional heads can specialize through their learned attention distributions while sharing one feature representation of digit content, with a single learned scalar retaining relative head weighting.

INTENDED_EDIT: Adopt the verified 1,062-parameter scalar MLP bias and triplet-tied attention bias, then replace the second head’s seven-coordinate value adapter with one learned head-level gain. This challenges the prior assumption that each routing head needs feature-wise value specialization.

EVIDENCE: Reference Design 1 achieved 99.94% with 1,062 parameters, and the 1,087-parameter shared-value design achieved 99.98%; since addition is operand-symmetric and the heads retain independent learned relative-position biases, routing—not coordinate-wise value scaling—is plausibly the load-bearing specialization.

<<<<<<< SEARCH
class TripletQuartetTiedBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with a learned bias triplet and quartet."""

    def __init__(self, in_features: int, out_features: int):
        if out_features != 8:
            raise ValueError("out_features must be eight")
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat(
            (self.free_bias[:1].expand(3), self.free_bias[1:].expand(4))
        )
        return F.linear(x, self.linear.weight, bias) @ self.basis.transpose(0, 1)
=======
class ScalarBiasMeanZeroLinear(MeanZeroLinear):
    """Mean-zero linear map with one scalar shared across bias coordinates."""

    def __init__(self, in_features: int, out_features: int):
        super().__init__(in_features, out_features)
        self.linear.bias = None
        self.free_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = self.free_bias.expand(self.linear.out_features)
        return F.linear(x, self.linear.weight, bias) @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim)
        )
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 2))
=======
        self.secondary_value_gain = nn.Parameter(torch.ones(n_head - 1))
        self.register_buffer("q_bias", torch.full((1,), 0.02), persistent=False)
        self.output_bias = nn.Parameter(torch.zeros(d_model - 3))
>>>>>>> REPLACE

<<<<<<< SEARCH
        value_gain = torch.cat(
            (
                y.new_ones(1, self.output_dim),
                self.secondary_value_gain,
            ),
            dim=0,
        )
=======
        value_gain = torch.cat(
            (
                y.new_ones(1, self.output_dim),
                self.secondary_value_gain[:, None].expand(-1, self.output_dim),
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat((self.output_bias[:-1], self.output_bias[-1:].expand(2)))
=======
        bias = torch.cat((self.output_bias[:-1], self.output_bias[-1:].expand(3)))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = TripletQuartetTiedBiasMeanZeroLinear(d_ff, d_model)
=======
        self.fc2 = ScalarBiasMeanZeroLinear(d_ff, d_model)
>>>>>>> REPLACE
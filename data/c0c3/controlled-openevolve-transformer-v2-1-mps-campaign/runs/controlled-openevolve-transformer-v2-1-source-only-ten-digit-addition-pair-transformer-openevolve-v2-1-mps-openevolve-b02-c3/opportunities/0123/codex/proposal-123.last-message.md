MECHANISM: Headwise query-coordinate gauge fixing

HYPOTHESIS: A 671-parameter transformer will retain at least 99% accuracy because it preserves the qualified 672-parameter architecture while fixing one query-bias coordinate that can be compensated through the learned second-head Q/K coordinate basis.

INTENDED_EDIT: Reconstruct the qualified four-coordinate MLP output-bias fixing, then remove the final second-head query-bias scalar by fixing it at zero.

EVIDENCE: The 672-parameter design achieved 99.52%; eliminating its remaining MLP output bias, tying a lexical coordinate, or fixing another final-normalization scale failed, motivating an isolated reduction in the previously untested headwise query-coordinate gauge while preserving those components.

<<<<<<< SEARCH
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        if out_features < 2:
            raise ValueError("out_features must be at least two")

        self.coeff = nn.Linear(in_features, out_features - 1, bias=bias)
=======
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        fixed_bias_coeffs: int = 0,
    ):
        super().__init__()
        if out_features < 2:
            raise ValueError("out_features must be at least two")
        if fixed_bias_coeffs < 0 or fixed_bias_coeffs >= out_features - 1:
            raise ValueError("invalid number of fixed bias coefficients")

        self.fixed_bias_coeffs = fixed_bias_coeffs
        self.coeff = nn.Linear(in_features, out_features - 1, bias=False)
        self.bias = (
            nn.Parameter(torch.zeros(out_features - 1 - fixed_bias_coeffs))
            if bias
            else None
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.coeff(x) @ self.basis.transpose(0, 1)
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = (
            None
            if self.bias is None
            else F.pad(self.bias, (0, self.fixed_bias_coeffs))
        )
        return F.linear(x, self.coeff.weight, bias) @ self.basis.transpose(0, 1)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q = q + self.q_bias
=======
        q = q + F.pad(self.q_bias, (0, 1))
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.fc2 = QuotientOutputLinear(d_ff, d_model)
=======
        self.fc2 = QuotientOutputLinear(
            d_ff, d_model, fixed_bias_coeffs=4
        )
>>>>>>> REPLACE
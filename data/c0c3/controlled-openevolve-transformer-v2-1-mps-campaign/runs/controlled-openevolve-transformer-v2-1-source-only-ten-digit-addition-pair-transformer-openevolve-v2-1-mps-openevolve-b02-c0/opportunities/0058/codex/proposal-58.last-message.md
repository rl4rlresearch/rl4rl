MECHANISM: Incremental value-bias/output-bias gauge elimination

HYPOTHESIS: Tying one additional value-projection bias coordinate will reduce the model from 1,615 to 1,614 parameters while retaining at least 99% accuracy, because a value bias contributes a position-independent attention output that can be absorbed by the learned attention output bias modulo common-mode shifts removed by downstream LayerNorms.

INTENDED_EDIT: Store six independent value-bias coordinates instead of seven, preserve the existing final-to-first tie, and reconstruct the penultimate coordinate by tying it to the second coordinate.

EVIDENCE: Extending the `fc2` gauge to a second column collapsed to 49.55%, whereas the distinct attention output-bias gauge previously retained 99.89% accuracy; this motivates testing a one-parameter redundancy coupled directly to that successful output bias rather than further constraining `fc2`.

<<<<<<< SEARCH
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with tied key and value bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-9].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_key_bias = self.bias[:1]
        value_bias = torch.cat(
            (self.bias[self.d_model :], self.bias[self.d_model : self.d_model + 1])
        )
=======
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with tied key and value bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-10].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_key_bias = self.bias[:1]
        value_bias = torch.cat(
            (
                self.bias[self.d_model :],
                self.bias[self.d_model + 1 : self.d_model + 2],
                self.bias[self.d_model : self.d_model + 1],
            )
        )
>>>>>>> REPLACE
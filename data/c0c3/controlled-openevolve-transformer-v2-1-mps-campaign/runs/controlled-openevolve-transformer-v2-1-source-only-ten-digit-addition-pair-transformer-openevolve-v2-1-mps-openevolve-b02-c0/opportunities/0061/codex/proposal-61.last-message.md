MECHANISM: Incremental symmetric value-bias gauge elimination

HYPOTHESIS: Tying the fourth independent value-projection bias coordinate to the third will reduce the model from 1,612 to 1,611 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store three independent value-bias coordinates and reconstruct the eight-coordinate bias as `[a, b, c, c, c, c, b, a]`.

EVIDENCE: The three consecutive symmetric value-bias reductions retained 99.90%, 99.93%, and 99.98% accuracy at 1,614, 1,613, and 1,612 parameters; extending the same successful tying pattern by one coordinate is the closest controlled next test.

<<<<<<< SEARCH
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with tied key and value bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-12].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_key_bias = self.bias[:1]
        value_bias = torch.cat(
            (
                self.bias[self.d_model :],
                self.bias[self.d_model + 3 : self.d_model + 4],
                self.bias[self.d_model + 2 : self.d_model + 3],
                self.bias[self.d_model + 1 : self.d_model + 2],
                self.bias[self.d_model : self.d_model + 1],
            )
        )
=======
class KeyAnchoredLinear(nn.Linear):
    """Combined QKV projection with tied key and symmetric value bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__(d_model, 3 * d_model)
        self.d_model = d_model
        self.bias = nn.Parameter(self.bias.detach()[:-13].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_key_bias = self.bias[:1]
        value_bias = torch.cat(
            (
                self.bias[self.d_model :],
                self.bias[self.d_model + 2 : self.d_model + 3],
                self.bias[self.d_model + 2 : self.d_model + 3],
                self.bias[self.d_model + 2 : self.d_model + 3],
                self.bias[self.d_model + 1 : self.d_model + 2],
                self.bias[self.d_model : self.d_model + 1],
            )
        )
>>>>>>> REPLACE
MECHANISM: Incremental symmetric value-bias gauge elimination

HYPOTHESIS: Tying the fifth value-projection bias coordinate to the fourth will reduce the model from 1,613 to 1,612 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store four independent value-bias coordinates and reconstruct the eight-coordinate bias symmetrically as `[a, b, c, d, d, c, b, a]`.

EVIDENCE: The two immediately preceding symmetric value-bias ties retained 99.90% and 99.93% accuracy at 1,614 and 1,613 parameters; extending the same parameterization by one coordinate is the closest controlled next test.

<<<<<<< SEARCH
        self.bias = nn.Parameter(self.bias.detach()[:-11].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_key_bias = self.bias[:1]
        value_bias = torch.cat(
            (
                self.bias[self.d_model :],
                self.bias[self.d_model + 2 : self.d_model + 3],
                self.bias[self.d_model + 1 : self.d_model + 2],
                self.bias[self.d_model : self.d_model + 1],
            )
        )
=======
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
>>>>>>> REPLACE
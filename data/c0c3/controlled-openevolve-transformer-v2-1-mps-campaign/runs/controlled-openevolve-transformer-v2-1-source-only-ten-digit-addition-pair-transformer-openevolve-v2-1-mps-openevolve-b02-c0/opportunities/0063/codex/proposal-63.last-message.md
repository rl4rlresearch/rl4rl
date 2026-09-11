MECHANISM: Expanded central value-bias anchoring

HYPOTHESIS: Fixing the next mirrored value-bias pair to zero will reduce the model from 1,611 to 1,610 parameters while retaining at least 99% accuracy, because value bias contributes a position-independent attention output absorbable by the learned output bias.

INTENDED_EDIT: Store two independent value-bias coordinates and reconstruct the eight-coordinate bias as `[a, b, 0, 0, 0, 0, b, a]`.

EVIDENCE: Anchoring the central mirrored pair retained 99.97% accuracy at 1,611 parameters, whereas conflating it with another coordinate collapsed accuracy; expanding the zero-anchored center tests another reduction without introducing destructive coordinate tying.

<<<<<<< SEARCH
        self.bias = nn.Parameter(self.bias.detach()[:-13].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_key_bias = self.bias[:1]
        value_bias = torch.cat(
            (
                self.bias[self.d_model :],
                self.bias.new_zeros(2),
                self.bias[self.d_model + 2 : self.d_model + 3],
                self.bias[self.d_model + 1 : self.d_model + 2],
                self.bias[self.d_model : self.d_model + 1],
            )
        )
=======
        self.bias = nn.Parameter(self.bias.detach()[:-14].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_key_bias = self.bias[:1]
        value_bias = torch.cat(
            (
                self.bias[self.d_model :],
                self.bias.new_zeros(4),
                self.bias[self.d_model + 1 : self.d_model + 2],
                self.bias[self.d_model : self.d_model + 1],
            )
        )
>>>>>>> REPLACE
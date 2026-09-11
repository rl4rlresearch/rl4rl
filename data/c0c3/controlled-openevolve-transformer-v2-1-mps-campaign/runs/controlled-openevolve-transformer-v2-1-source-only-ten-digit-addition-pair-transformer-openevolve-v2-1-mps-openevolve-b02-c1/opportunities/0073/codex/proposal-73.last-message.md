MECHANISM: Reference-anchored final-scale interpolation

HYPOTHESIS: Reconstructing the shared final LayerNorm scale pair as the midpoint between the preceding learned scale and the fixed residual-scale reference will reduce the model from 1,002 to 1,001 parameters while retaining at least 99% accuracy.

INTENDED_EDIT: Store five final LayerNorm scale coordinates and derive the sixth and seventh coordinates from a learned-to-reference midpoint; leave all other model and training behavior unchanged.

EVIDENCE: Sharing the last two learned scales achieved 99.95%, while extending equality across three scales reached 96.84%; midpoint reconstruction preserves the successful pair sharing while avoiding the failed flat triplet.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(normalized_shape - 2))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            [
                self.weight,
                self.weight[-1:],
                self.weight.new_ones(1),
            ]
        )
=======
        self.weight = nn.Parameter(torch.ones(normalized_shape - 3))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        reference = self.weight.new_ones(1)
        boundary_scale = 0.5 * (self.weight[-1:] + reference)
        weight = torch.cat(
            [
                self.weight,
                boundary_scale.expand(2),
                reference,
            ]
        )
>>>>>>> REPLACE
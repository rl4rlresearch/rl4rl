MECHANISM: Learned adjacent final-normalization scale triplet sharing

HYPOTHESIS: Sharing the final LayerNorm’s last three learned scale coordinates will reduce the model from 1,039 to 1,038 parameters while retaining at least 99% accuracy, because sharing the last two scales achieved 99.95% and this removes only one adjacent scale degree of freedom.

INTENDED_EDIT: Store five final LayerNorm scale coordinates and reconstruct the sixth and seventh learned coordinates from the final stored scale, while retaining the eighth coordinate as the fixed residual-scale reference.

EVIDENCE: The immediately prior adjacent scale-sharing result achieved 99.95% at 1,039 parameters, whereas adjacent final-normalization bias sharing reached only 82.53%; this supports extending compression in the demonstrably tolerant scale pathway.

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
        weight = torch.cat(
            [
                self.weight,
                self.weight[-1:].expand(2),
                self.weight.new_ones(1),
            ]
        )
>>>>>>> REPLACE
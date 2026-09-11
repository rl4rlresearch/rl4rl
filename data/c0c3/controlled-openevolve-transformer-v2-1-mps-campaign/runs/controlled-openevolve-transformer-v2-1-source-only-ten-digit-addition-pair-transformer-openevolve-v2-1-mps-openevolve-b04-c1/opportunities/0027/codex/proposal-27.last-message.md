MECHANISM: Second first-LayerNorm scale gauge fixing

HYPOTHESIS: Fixing a second first-LayerNorm scale coordinate to one will reduce the model from 1607 to 1606 parameters while retaining at least 99% accuracy, because both fixed scales can be absorbed independently into the corresponding dense QKV projection columns.

INTENDED_EDIT: Store six learned scales in `ScaleFixedLayerNorm` and append two fixed unit scales, preserving the initial function and all verified attention capacities.

EVIDENCE: Fixing one first-LayerNorm scale achieved 99.92% accuracy with 1607 parameters; extending that successful gauge fixing within the same LayerNorm is more directly supported than the failed second-LayerNorm ablation.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
=======
        self.weight = nn.Parameter(torch.ones(d_model - 2))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(2)))
>>>>>>> REPLACE
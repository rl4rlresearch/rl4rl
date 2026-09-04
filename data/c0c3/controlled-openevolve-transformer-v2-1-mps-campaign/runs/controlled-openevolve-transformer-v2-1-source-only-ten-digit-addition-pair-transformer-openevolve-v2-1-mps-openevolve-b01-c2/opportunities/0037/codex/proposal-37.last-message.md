MECHANISM: Incremental attention LayerNorm scale anchoring

HYPOTHESIS: Fixing a fourth `ln1` scale coordinate in the verified 1,589-parameter design will produce a 1,588-parameter model with at least 99% accuracy.

INTENDED_EDIT: Retain four learned `ln1` scales and anchor the remaining four coordinates to one.

EVIDENCE: Anchoring three `ln1` scales achieved 99.73% accuracy with 1,589 parameters; anchoring one additional coordinate is the smallest incremental reduction supported by the successful two- and three-coordinate results.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(d_model - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(3)))
=======
        self.weight = nn.Parameter(torch.ones(d_model - 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(4)))
>>>>>>> REPLACE
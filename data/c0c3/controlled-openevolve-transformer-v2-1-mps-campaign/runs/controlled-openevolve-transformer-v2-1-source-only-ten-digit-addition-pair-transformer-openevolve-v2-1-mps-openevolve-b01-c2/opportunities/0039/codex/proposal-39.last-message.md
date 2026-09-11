MECHANISM: Sixth-coordinate attention LayerNorm scale anchoring

HYPOTHESIS: Anchoring a sixth `ln1` scale coordinate in the verified 1,587-parameter design will produce a 1,586-parameter model while maintaining at least 99% accuracy.

INTENDED_EDIT: Retain two learned `ln1` scales and fix the remaining six coordinates to one.

EVIDENCE: Reference Design 2 retained three learned scales and achieved 99.40% accuracy with 1,587 parameters; anchoring one additional coordinate is the smallest untested reduction along the successful incremental scale-anchoring path.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
=======
        self.weight = nn.Parameter(torch.ones(d_model - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(6)))
>>>>>>> REPLACE
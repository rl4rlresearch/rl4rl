MECHANISM: Fifth-coordinate attention LayerNorm scale anchoring

HYPOTHESIS: Anchoring a fifth `ln1` scale coordinate will reduce the model to 1,587 parameters while maintaining at least 99% accuracy.

INTENDED_EDIT: Retain three learned `ln1` scales and fix the remaining five coordinates to one.

EVIDENCE: Reference Design 1 anchored four coordinates and achieved 99.90% accuracy with 1,588 parameters; anchoring one additional coordinate is the smallest incremental reduction, while the failed affine-free result cautions against removing all scales at once.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(2)))
=======
        self.weight = nn.Parameter(torch.ones(d_model - 5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(5)))
>>>>>>> REPLACE
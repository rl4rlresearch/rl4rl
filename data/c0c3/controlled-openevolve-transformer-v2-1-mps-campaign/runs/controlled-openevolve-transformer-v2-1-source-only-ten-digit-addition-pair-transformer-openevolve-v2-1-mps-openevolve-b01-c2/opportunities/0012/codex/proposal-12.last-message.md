MECHANISM: Second-coordinate LayerNorm scale gauge fixing

HYPOTHESIS: Fixing a second `ln2` scale coordinate at one will reduce the model from 1,611 to 1,610 learned parameters while retaining at least 99% accuracy, because the successful first anchor achieved 100% accuracy and the following MLP input weights can absorb both fixed coordinate scales.

INTENDED_EDIT: Retain six learned `ln2` scales and synthesize two fixed unit scales.

EVIDENCE: The 1,611-parameter model with one fixed `ln2` scale coordinate achieved 100% accuracy, while removing all normalization scales failed; extending the successful reduction by exactly one coordinate is the most conservative capacity probe.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(d_model - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
=======
        self.weight = nn.Parameter(torch.ones(d_model - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(2)))
>>>>>>> REPLACE
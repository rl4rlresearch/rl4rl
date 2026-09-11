MECHANISM: Fifth pre-attention LayerNorm bias/projection-bias redundancy

HYPOTHESIS: Removing a fifth `ln1` bias degree of freedom will reduce the model from 1616 to 1615 parameters while retaining at least 99% accuracy, because its additive effect remains absorbable by the query and value/output bias paths and is softmax-null for keys.

INTENDED_EDIT: Store three learned `ln1` bias contrasts instead of four and reconstruct four anchored coordinates before centering the seven active coordinates.

EVIDENCE: Removing the fourth `ln1` bias degree achieved 99.46% accuracy at 1616 parameters, so extending the same pathway-specific mechanism by one coordinate is the smallest evidence-backed next reduction.

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.zeros(d_model - 4))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_bias = torch.cat((self.bias, self.bias.new_zeros(3)))
=======
        self.bias = nn.Parameter(torch.zeros(d_model - 5))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_bias = torch.cat((self.bias, self.bias.new_zeros(4)))
>>>>>>> REPLACE
MECHANISM: Fourth pre-MLP LayerNorm bias/input-bias redundancy

HYPOTHESIS: Removing a fourth `ln2` bias degree of freedom will reduce the model from 1618 to 1617 parameters while retaining at least 99% accuracy, because its zero-initialized additive contribution remains representable by the downstream `fc1` bias.

INTENDED_EDIT: Store four learned `ln2` bias contrasts instead of five and reconstruct three anchored coordinates before centering the seven active coordinates.

EVIDENCE: Removing the third `ln2` bias degree achieved 99.93% accuracy at 1618 parameters, while fixing an `ln2` scale achieved only 73.64%; this supports extending the successful additive-bias reduction by one coordinate.

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.zeros(d_model - 3))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_bias = torch.cat((self.bias, self.bias.new_zeros(2)))
=======
        self.bias = nn.Parameter(torch.zeros(d_model - 4))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_bias = torch.cat((self.bias, self.bias.new_zeros(3)))
>>>>>>> REPLACE
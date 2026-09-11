MECHANISM: Final untested pre-MLP scale gauge fixing

HYPOTHESIS: Fixing pre-MLP LayerNorm gain coordinate 0 to one will reduce the model from 1609 to 1608 parameters while retaining at least 99% accuracy, because its scale is absorbable by column 0 of `fc1.weight` and its bias has remained successfully fixed throughout every passing anchored design.

INTENDED_EDIT: Remove gain coordinate 0 from the learned vector and reconstruct gains 0 and 3 through 7 as fixed ones, leaving the sensitive gains 1 and 2 learned.

EVIDENCE: Fixing gains 3 through 7 retained 100% accuracy at 1611 parameters, while fixing gains 1 or 2 collapsed to 27.21% and 48.42%; coordinate 0 is the only untested gain and already tolerates a fixed LayerNorm bias in the 99.99%-accurate 1609-parameter design.

<<<<<<< SEARCH
        self.weight_rest = nn.Parameter(torch.ones(d_model - 5))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat((self.weight_rest, ones(5)))
=======
        self.weight_rest = nn.Parameter(torch.ones(d_model - 6))
        self.bias_rest = nn.Parameter(torch.zeros(d_model - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        zeros = self.bias_rest.new_zeros
        ones = self.weight_rest.new_ones
        weight = torch.cat((ones(1), self.weight_rest, ones(5)))
>>>>>>> REPLACE
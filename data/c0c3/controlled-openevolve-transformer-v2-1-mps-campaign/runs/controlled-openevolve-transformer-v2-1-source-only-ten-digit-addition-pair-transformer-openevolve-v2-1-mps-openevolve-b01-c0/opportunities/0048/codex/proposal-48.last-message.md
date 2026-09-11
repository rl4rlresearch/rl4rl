MECHANISM: Coordinate-swapped sixth pre-attention LayerNorm bias reduction

HYPOTHESIS: Reducing `ln1` to two learned bias contrasts will retain at least 99% accuracy when the previously discarded third active coordinate is preserved instead of the second, indicating that the 96.77% result was coordinate-specific rather than an intrinsic two-degree capacity limit.

INTENDED_EDIT: Store two `ln1` bias parameters and place them in the first and third active coordinates, fixing the second and remaining coordinates before centering.

EVIDENCE: The prior sixth `ln1` reduction reached 96.77%, substantially closer to the requirement than other one-parameter reductions that collapsed to 16.94–74.01%; testing a minimally changed contrast basis is therefore the most informative next 1614-parameter design.

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.zeros(d_model - 5))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_bias = torch.cat((self.bias, self.bias.new_zeros(4)))
        active_bias = active_bias - active_bias.mean()
=======
        self.bias = nn.Parameter(torch.zeros(d_model - 6))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_bias = torch.cat(
            (
                self.bias[:1],
                self.bias.new_zeros(1),
                self.bias[1:],
                self.bias.new_zeros(4),
            )
        )
        active_bias = active_bias - active_bias.mean()
>>>>>>> REPLACE
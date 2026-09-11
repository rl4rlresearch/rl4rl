MECHANISM: Balanced LayerNorm-gain contrast gauge

HYPOTHESIS: Constraining the eight `ln2` gains to seven learned mean-one contrasts will reduce the model from 1634 to 1633 parameters while retaining at least 99% accuracy, because every effective gain remains trainable and the removed common-scale direction is locally absorbable by `fc1.weight` and `fc1.bias`.

INTENDED_EDIT: Replace the eight independent `ln2` gain parameters with seven parameters that reconstruct eight mean-one gains, preserving the initial computation exactly.

EVIDENCE: Centering six learned contrasts across seven active `ln2` bias coordinates achieved 99.96% at 1634 parameters after a fixed-coordinate reduction failed; this motivates applying the same balanced parameterization to the locally redundant common-gain direction instead of fixing one gain coordinate.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, self.bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = nn.Parameter(torch.zeros(d_model - 2))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        weight = weight - weight.mean() + 1.0
        active_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, self.bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
>>>>>>> REPLACE
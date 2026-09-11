MECHANISM: Balanced LayerNorm-bias contrast gauge

HYPOTHESIS: Replacing the failed second fixed-coordinate ablation with six learned zero-mean contrasts across the seven active `ln2` bias coordinates will reduce the model to 1634 parameters while retaining at least 99% accuracy, because every active coordinate remains trainable and the removed common-offset direction is locally absorbable by `fc1.bias`.

INTENDED_EDIT: Parameterize the first seven `ln2` bias coordinates as six centered contrasts while keeping the eighth coordinate fixed at zero.

EVIDENCE: One fixed `ln2` bias coordinate achieved 99.87% at 1635 parameters, but fixing an adjacent second coordinate fell to 76.37%; this motivates distributing the second constraint across all seven active coordinates instead of eliminating another coordinate’s optimization path.

<<<<<<< SEARCH
        self.bias = nn.Parameter(torch.zeros(d_model - 1))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
        self.bias = nn.Parameter(torch.zeros(d_model - 2))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_bias = torch.cat((self.bias, self.bias.new_zeros(1)))
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, self.bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE
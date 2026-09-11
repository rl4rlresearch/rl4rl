MECHANISM: Bias-free pre-attention LayerNorm

HYPOTHESIS: Removing the final learned `ln1` bias contrast will reduce the model to 1612 parameters while retaining at least 99% accuracy, because the additive offset remains absorbable by the attention projection biases and softmax-null key pathway.

INTENDED_EDIT: Eliminate the third-coordinate `ln1` bias parameter and use a fixed zero bias in the pre-attention LayerNorm.

EVIDENCE: The third-coordinate-only `ln1` design achieved 99.90% accuracy at 1613 parameters, after successive reductions from five contrasts continued to pass; removing its final contrast is the smallest and most informative continuation of that successful pathway.

<<<<<<< SEARCH
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        active_bias = torch.cat(
            (
                self.bias.new_zeros(2),
                self.bias,
                self.bias.new_zeros(4),
            )
        )
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, self.bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = x.new_zeros(self.normalized_shape)
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE
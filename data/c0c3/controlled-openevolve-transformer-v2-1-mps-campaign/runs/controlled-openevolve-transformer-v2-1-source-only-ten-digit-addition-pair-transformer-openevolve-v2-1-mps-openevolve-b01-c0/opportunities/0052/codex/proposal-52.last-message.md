MECHANISM: Shared pre-attention/query-bias gauge

HYPOTHESIS: Tying the remaining third-coordinate `ln1` bias contrast to the third query-bias parameter will reduce the model to 1612 parameters while retaining at least 99% accuracy, because the contrast remains learned and its projection-space effect is gauge-coupled to the query/value biases.

INTENDED_EDIT: Remove the standalone `ln1` bias parameter and reuse `attn.qv_bias[2]` as its learned third-coordinate contrast.

EVIDENCE: The standalone third-coordinate contrast achieved 99.90% at 1613 parameters, whereas fixing it to zero reached 97.43%; sharing it with an existing zero-initialized query bias tests whether retaining an adaptive contrast, rather than an independent degree of freedom, supplies the optimization benefit.

<<<<<<< SEARCH
class CenteredBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
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
class CenteredBiasLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = 1e-5

    def forward(
        self, x: torch.Tensor, shared_bias: torch.Tensor
    ) -> torch.Tensor:
        active_bias = torch.cat(
            (
                shared_bias.new_zeros(2),
                shared_bias.reshape(1),
                shared_bias.new_zeros(4),
            )
        )
        active_bias = active_bias - active_bias.mean()
        bias = torch.cat((active_bias, shared_bias.new_zeros(1)))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x
=======
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x, self.attn.qv_bias[2]))
        x = x + self.mlp(self.ln2(x))
        return x
>>>>>>> REPLACE
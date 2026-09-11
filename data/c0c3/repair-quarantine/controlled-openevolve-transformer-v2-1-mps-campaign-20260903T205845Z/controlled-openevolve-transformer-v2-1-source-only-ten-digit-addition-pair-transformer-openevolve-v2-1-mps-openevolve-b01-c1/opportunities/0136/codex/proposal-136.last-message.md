MECHANISM: Adaptive mean-tied attention LayerNorm bias

HYPOTHESIS: Tying `ln1` bias coordinate three to the mean of coordinates zero through two will reduce the model to 1,524 parameters while retaining at least 99% accuracy by restoring adaptive dynamics absent when that coordinate was fixed at zero.

INTENDED_EDIT: Store three independent `ln1` bias values, reconstruct coordinate three as their mean, and keep coordinates four through seven fixed at zero.

EVIDENCE: Fixing `ln1` bias coordinate three scored 98.13%, while adaptively mean-tying the essential `ln2` coordinate six improved its fixed-value result from 66.5% to 97.73%; the same adaptive sharing may recover the smaller remaining gap here.

<<<<<<< SEARCH
class FourPrunedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with four bias coordinates absorbed by downstream attention."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-4].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        bias = F.pad(self.bias, (0, 4))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
=======
class MeanTiedBiasLayerNorm(nn.LayerNorm):
    """LayerNorm with four absorbed biases and one adaptively shared bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        full_bias = self.bias.detach()
        self.bias = nn.Parameter(full_bias[:-5].clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shared_bias = self.bias.mean().unsqueeze(0)
        bias = F.pad(torch.cat((self.bias, shared_bias)), (0, 4))
        return F.layer_norm(x, self.normalized_shape, self.weight, bias, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = FourPrunedBiasLayerNorm(cfg.d_model)
=======
        self.ln1 = MeanTiedBiasLayerNorm(cfg.d_model)
>>>>>>> REPLACE
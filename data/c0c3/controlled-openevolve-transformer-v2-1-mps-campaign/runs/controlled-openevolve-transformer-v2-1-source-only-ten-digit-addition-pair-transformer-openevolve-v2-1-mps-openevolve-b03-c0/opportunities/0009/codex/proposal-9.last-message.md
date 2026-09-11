MECHANISM: Single-coordinate final-normalization scale sharing

HYPOTHESIS: Sharing one of eight `ln_f` scale coordinates with the block LayerNorm will reduce deduplicated parameters from 1,620 to 1,619 while retaining at least 99% accuracy.

INTENDED_EDIT: Replace the final LayerNorm with an equivalent learned normalization whose first scale coordinate and bias are shared with `blocks[0].ln1`, while its other seven scale coordinates remain independent.

EVIDENCE: Sharing only the final bias achieved 99.40% at 1,620 parameters, whereas sharing all eight final scale coordinates collapsed accuracy to 27.07%; a one-coordinate constraint is the smallest informative step toward locating that capacity boundary.

<<<<<<< SEARCH
class Block(nn.Module):
=======
class PartiallySharedLayerNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        shared_weight: nn.Parameter,
        shared_bias: nn.Parameter,
    ):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.shared_weight = shared_weight
        self.weight_tail = nn.Parameter(torch.ones(d_model - 1))
        self.bias = shared_bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.shared_weight[:1], self.weight_tail))
        return F.layer_norm(x, self.normalized_shape, weight, self.bias, 1e-5)


class Block(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.ln_f.bias = self.blocks[0].ln1.bias
=======
        self.ln_f = PartiallySharedLayerNorm(
            cfg.d_model,
            self.blocks[0].ln1.weight,
            self.blocks[0].ln1.bias,
        )
>>>>>>> REPLACE
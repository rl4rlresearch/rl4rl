MECHANISM: Independent pre-MLP dynamic LayerNorm scale tying

HYPOTHESIS: Tying the last `ln2` scale coordinate to its first learned coordinate will reduce the model from 1,632 to 1,631 parameters while retaining at least 99% accuracy, because `fc1` can absorb independent scale differences while the tied scale remains dynamically learned.

INTENDED_EDIT: Add a scale-tied LayerNorm that preserves all eight bias coordinates but learns seven scale coordinates, and use it for `ln2`.

EVIDENCE: A single dynamic `ln1` scale tie achieved 99.78% accuracy, whereas a second simultaneous scale tie failed and fixing an `ln2` bias coordinate achieved only 13.53%; this motivates one learned scale tie in the distinct pre-MLP normalization while leaving every `ln2` bias trainable.

<<<<<<< SEARCH
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)


class Block(nn.Module):
=======
        return F.layer_norm(x, (x.size(-1),), weight, bias, 1e-5)


class ScaleTiedLayerNorm(nn.Module):
    """LayerNorm with one dynamically tied scale coordinate."""

    def __init__(self, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight[:1]))
        return F.layer_norm(x, (x.size(-1),), weight, self.bias, 1e-5)


class Block(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = ScaleTiedLayerNorm(cfg.d_model)
>>>>>>> REPLACE
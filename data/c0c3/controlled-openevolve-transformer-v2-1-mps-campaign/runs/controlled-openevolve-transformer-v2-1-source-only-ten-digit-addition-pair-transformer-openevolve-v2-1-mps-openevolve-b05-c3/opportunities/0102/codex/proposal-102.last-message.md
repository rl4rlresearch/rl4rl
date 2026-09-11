MECHANISM: Four-way final-LayerNorm scale tying

HYPOTHESIS: Extending the qualified three-way final-scale tie to four channels will reduce the model from 1,250 to 1,249 parameters while retaining at least 99% accuracy, because five independent final scales and the shared adaptive pre-attention signal remain.

INTENDED_EDIT: Add a final LayerNorm reconstructed from five learned scales with its last four channel scales tied, and derive the shared `ln1` scale from the reconstructed full scale vector.

EVIDENCE: The three-way final-scale tie achieved 99.84% accuracy with 1,250 parameters, after the two-way tie achieved 99.77%; this supports testing one further isolated final-scale tie instead of reducing the routing kernel whose fifth terminal tie collapsed to 91.63%.

<<<<<<< SEARCH
    def full_weight(self) -> torch.Tensor:
        shared_scale = self._scale_source.weight.mean().reshape(1)
        return torch.cat((shared_scale, self.fixed_weight))
=======
    def full_weight(self) -> torch.Tensor:
        source_weight = self._scale_source.full_weight()
        shared_scale = source_weight.mean().reshape(1)
        return torch.cat((shared_scale, self.fixed_weight))
>>>>>>> REPLACE

<<<<<<< SEARCH
class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
=======
class TiedFinalScaleLayerNorm(nn.LayerNorm):
    """Final LayerNorm with its last four channel scales tied."""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.weight = nn.Parameter(torch.ones(d_model - 3))

    def full_weight(self) -> torch.Tensor:
        return torch.cat((self.weight, self.weight[-1:].expand(3)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.full_weight(),
            self.bias,
            self.eps,
        )


class GaugeFixedMLPScaleLayerNorm(nn.LayerNorm):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = TiedFinalScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE
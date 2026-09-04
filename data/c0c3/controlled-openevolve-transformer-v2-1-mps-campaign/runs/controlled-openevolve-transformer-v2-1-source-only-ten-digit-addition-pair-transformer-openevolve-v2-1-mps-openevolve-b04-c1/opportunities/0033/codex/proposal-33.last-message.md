MECHANISM: Isotropic second-LayerNorm scale gauge

HYPOTHESIS: Reparameterizing the second LayerNorm’s eight scales as unit scales plus seven orthonormal zero-mean coordinates will reduce the model from 1605 to 1604 parameters while retaining at least 99% accuracy, because its omitted common scale is absorbable by the dense `fc1` projection without selecting a privileged feature coordinate.

INTENDED_EDIT: Add a seven-parameter, symmetry-preserving LayerNorm scale parameterization and use it for `ln2`, leaving the verified first LayerNorm and all attention and MLP capacities unchanged.

EVIDENCE: Fixing one `ln2` coordinate failed at 20.74%, but replacing an asymmetric MLP-bias anchor with an isotropic Householder quotient recovered 99.92% accuracy at 1605 parameters; this applies the same corrective parameterization to the previously unsuccessful LayerNorm gauge.

<<<<<<< SEARCH
    def forward(self) -> torch.Tensor:
        bias = F.pad(self.coordinates, (1, 0))
        projection = 2.0 * torch.dot(self.reflector, bias) / self.reflector_norm_sq
        return bias - self.reflector * projection


class ShiftGaugeEmbedding(nn.Module):
=======
    def forward(self) -> torch.Tensor:
        bias = F.pad(self.coordinates, (1, 0))
        projection = 2.0 * torch.dot(self.reflector, bias) / self.reflector_norm_sq
        return bias - self.reflector * projection


class ScaleGaugeLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.scale_offset = ShiftGaugeBias(d_model)
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = 1.0 + self.scale_offset()
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)


class ShiftGaugeEmbedding(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model, bias=False)
=======
        self.ln2 = ScaleGaugeLayerNorm(cfg.d_model)
>>>>>>> REPLACE
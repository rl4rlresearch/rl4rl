MECHANISM: Optimizer-preserving final-LayerNorm bias gauge quotient

HYPOTHESIS: Fixing one common-mode coordinate of the final LayerNorm bias will reduce the model from 1,284 to 1,283 parameters while retaining at least 99% accuracy, because the centered tied classifier exactly annihilates uniform hidden-state shifts.

INTENDED_EDIT: Store seven final-LayerNorm bias differences, reconstruct an eighth zero reference coordinate during inference, and preserve full-width AdamW moments through the existing quotient optimizer.

EVIDENCE: The 1,284-parameter row-centered tied-embedding model achieved 100% accuracy, and prior optimizer-preserving quotients retained at least 99.75% accuracy across exact residual and attention invariances.

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
=======
class GaugeFixedFinalLayerNorm(nn.Module):
    """Final LayerNorm modulo the bias's centered-out common mode."""

    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            F.pad(self.bias, (0, 1)),
            self.eps,
        )


class TinyDecoderLM(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = GaugeFixedFinalLayerNorm(cfg.d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.gauge_specs = [
            (model.token_emb.weight, 1)
        ] + [
=======
        self.gauge_specs = [
            (model.token_emb.weight, 1),
            (model.ln_f.bias, 0),
        ] + [
>>>>>>> REPLACE
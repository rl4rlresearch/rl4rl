MECHANISM: Fused scale-only terminal normalization

HYPOTHESIS: Replacing the final LayerNorm’s learned bias with a fixed zero buffer will reduce the model from 1,384 to 1,376 parameters while retaining at least 99% accuracy and completing the proven 52,000-step schedule because the forward pass still supplies a bias tensor to the optimized LayerNorm path.

INTENDED_EDIT: Add a scale-only LayerNorm whose zero bias is a non-parameter buffer, and use it for the terminal normalization without changing training settings.

EVIDENCE: The 1,384-parameter model reached 99.99% at 52,000 steps; every prior 1,376-parameter attempt failed only by timeout, including with sparse validation, so preserving the baseline LayerNorm execution path directly tests whether the no-bias kernel path caused the runtime regression.

<<<<<<< SEARCH
class TinyDecoderLM(nn.Module):
=======
class ScaleOnlyLayerNorm(nn.LayerNorm):
    """LayerNorm with learned scale and a fixed zero bias."""

    def __init__(self, normalized_shape: int):
        super().__init__(normalized_shape)
        bias = self.bias.detach()
        del self._parameters["bias"]
        self.register_buffer("bias", bias)


class TinyDecoderLM(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = ScaleOnlyLayerNorm(cfg.d_model)
>>>>>>> REPLACE
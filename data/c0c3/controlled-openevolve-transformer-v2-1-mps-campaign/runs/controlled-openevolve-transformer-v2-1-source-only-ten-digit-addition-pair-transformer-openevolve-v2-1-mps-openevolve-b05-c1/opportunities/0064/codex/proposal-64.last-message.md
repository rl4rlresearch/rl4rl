MECHANISM: Cross-layer reuse of the essential final normalization bias

HYPOTHESIS: Reusing `attn.proj.bias[1]` as the single final LayerNorm bias will reduce the model from 1,308 to 1,307 parameters while retaining at least 99% accuracy because it preserves a trainable, zero-initialized output shift without coupling it to the sensitive MLP.

INTENDED_EDIT: Remove the dedicated one-coordinate final LayerNorm bias and supply an otherwise unused attention projection-bias coordinate during the final normalization.

EVIDENCE: The dedicated final bias achieved 100% accuracy at 1,308 parameters, while fixing it to zero fell to 29.11%, showing that adaptivity is important; the failed third MLP bias sharing coupled the scalar directly into the sensitive MLP, which this output-only reuse avoids.

<<<<<<< SEARCH
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm with seven zero-fixed output-bias coordinates."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model - 7))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            F.pad(self.bias, (0, 7)),
        )
=======
class FixedBiasLayerNorm(nn.Module):
    """LayerNorm whose sole nonfixed bias coordinate is shared."""

    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor, shared_bias: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            F.pad(shared_bias.reshape(1), (0, 7)),
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        x = self.ln_f(x)
        logits = self.lm_head(x)
=======
        x = self.ln_f(x, self.blocks[0].attn.proj.bias[1])
        logits = self.lm_head(x)
>>>>>>> REPLACE
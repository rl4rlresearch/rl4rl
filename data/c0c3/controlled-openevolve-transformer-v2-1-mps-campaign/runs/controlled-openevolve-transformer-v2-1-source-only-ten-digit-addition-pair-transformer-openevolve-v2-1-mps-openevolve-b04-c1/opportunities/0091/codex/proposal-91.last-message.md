MECHANISM: Dead output-channel affine pruning

HYPOTHESIS: Fixing the final LayerNorm affine values on its last four channels will reduce the model from 895 to 887 parameters while retaining at least 99% accuracy, because the compact tied output embedding has exact zeros in those channels, so their learned scales and biases cannot affect any token logit.

INTENDED_EDIT: Replace the final full affine LayerNorm with a compact affine LayerNorm that learns only the four logit-visible channel scales and biases while preserving normalization across all eight channels.

EVIDENCE: The 895-parameter design achieved 99.92% accuracy with a four-channel token embedding padded by four exact zeros; unlike the failed MLP tying experiment, this removes only parameters annihilated by the existing output projection.

<<<<<<< SEARCH
class ScaleFixedLayerNorm(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.weight = nn.Parameter(torch.ones(d_model - 1))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(x, self.normalized_shape, weight, None, self.eps)
=======
class CompactOutputLayerNorm(nn.Module):
    """Learn affine values only for channels visible to the tied readout."""

    def __init__(self, d_model: int, output_dim: int):
        super().__init__()
        self.normalized_shape = (d_model,)
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.ones(output_dim))
        self.bias = nn.Parameter(torch.zeros(output_dim))
        self.eps = 1e-5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tail = self.normalized_shape[0] - self.output_dim
        weight = F.pad(self.weight, (0, tail), value=1.0)
        bias = F.pad(self.bias, (0, tail))
        return F.layer_norm(
            x, self.normalized_shape, weight, bias, self.eps
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = nn.LayerNorm(cfg.d_model)
=======
        self.ln_f = CompactOutputLayerNorm(
            cfg.d_model, self.token_emb.compact_dim
        )
>>>>>>> REPLACE
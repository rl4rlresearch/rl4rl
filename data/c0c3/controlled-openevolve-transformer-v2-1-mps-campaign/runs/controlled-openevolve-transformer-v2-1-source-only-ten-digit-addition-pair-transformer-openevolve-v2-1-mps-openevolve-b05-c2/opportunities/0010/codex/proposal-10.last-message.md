MECHANISM: One-coordinate pre-MLP LayerNorm scale anchoring

HYPOTHESIS: Fixing one `ln2` scale coordinate will reduce the qualified 1583-parameter design to 1582 parameters while retaining at least 99% accuracy, because that scale is exactly absorbable into the corresponding `fc1` input column without constraining the MLP function.

INTENDED_EDIT: Restore all eight learned value-bias coordinates, remove `ln2` bias as in Reference Design 3, and represent its scale with seven learned coordinates plus one fixed unit coordinate.

EVIDENCE: The full-value-bias, bias-free-`ln2` design achieved 99.96% accuracy with 1583 parameters, whereas the prior 1582-parameter combination failed after removing a sensitive value-bias coordinate; this instead removes an algebraically redundant LayerNorm scale coordinate.

<<<<<<< SEARCH
        return F.linear(x, weight, bias)


class CausalSelfAttention(nn.Module):
=======
        return F.linear(x, weight, bias)


class AnchoredScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with one scale absorbed into its following linear map."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat((self.weight, self.weight.new_ones(1)))
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)


class CausalSelfAttention(nn.Module):
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.v_bias = nn.Parameter(torch.zeros(d_model - 1))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.qkv.bias = None
        self.qv_bias = nn.Parameter(torch.zeros(2, d_model))
        self.proj = MeanZeroOutputLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        bsz, seqlen, d_model = x.shape
        value_bias = F.pad(self.v_bias, (0, 1))
        qkv_bias = torch.cat(
            (self.q_bias, self.qkv.weight.new_zeros(d_model), value_bias)
        )
=======
        bsz, seqlen, d_model = x.shape
        qkv_bias = torch.cat(
            (self.qv_bias[0], self.qkv.weight.new_zeros(d_model), self.qv_bias[1])
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = nn.LayerNorm(cfg.d_model)
=======
        self.ln2 = AnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE
MECHANISM: Cross-module query-bias/LayerNorm scale sharing

HYPOTHESIS: Fixing `q_bias[0]` as in the qualified 1511-parameter design while sharing its remaining learned `ln1` scale with an existing query-bias scalar will yield a 1510-parameter model with at least 99% accuracy.

INTENDED_EDIT: Reconstruct the first coordinate of both attention-head query biases as zero, fix seven `ln1` scales to one, and derive the remaining scale from an already-counted query-bias parameter.

EVIDENCE: The distinct query-bias reduction reached 99.96% at 1511 parameters, whereas fixing the final `ln1` scale outright fell to 74.13%; parameter sharing retains an adaptive scale while testing one further deduplicated reduction.

<<<<<<< SEARCH
        self.q_bias = nn.Parameter(torch.zeros(d_model - 1))
=======
        self.q_bias = nn.Parameter(torch.zeros(d_model - 2))
>>>>>>> REPLACE

<<<<<<< SEARCH
        q_bias = torch.cat(
            (
                self.q_bias[:self.head_dim],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim:],
            )
        )
=======
        q_bias = torch.cat(
            (
                self.q_bias.new_zeros(1),
                self.q_bias[:self.head_dim - 1],
                self.q_bias.new_zeros(1),
                self.q_bias[self.head_dim - 1:],
            )
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
class SextupleAnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with six scales fixed through the following linear-map gauge."""
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim - 6))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 6), value=1.0)
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)
=======
class QueryTiedScaleLayerNorm(nn.Module):
    """LayerNorm with seven fixed scales and one query-bias-tied scale."""
    def __init__(
        self,
        dim: int,
        attention: CausalSelfAttention,
        eps: float = 1e-5,
    ):
        super().__init__()
        self.dim = dim
        self.eps = eps
        object.__setattr__(self, "_attention", attention)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        source = self._attention.q_bias
        leading_scale = 1.0 + source[0]
        weight = torch.cat(
            (leading_scale.unsqueeze(0), source.new_ones(self.dim - 1))
        )
        return F.layer_norm(x, (self.dim,), weight, None, self.eps)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln1 = SextupleAnchoredScaleLayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
=======
        self.attn = CausalSelfAttention(cfg.d_model, cfg.n_head, cfg.dropout, cfg.max_seq_len)
        self.ln1 = QueryTiedScaleLayerNorm(cfg.d_model, self.attn)
>>>>>>> REPLACE
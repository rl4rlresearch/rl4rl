MECHANISM: Full headwise query-bias rotational gauge fixing

HYPOTHESIS: Reducing each head’s query bias from four coordinates to one scalar, while applying the verified anchored terminal scale, will produce a 1,528-parameter model with at least 99% accuracy because unrestricted query/key projections can rotate any nonzero headwise bias onto a single coordinate without changing attention logits.

INTENDED_EDIT: Use one learned query-bias scalar per head reconstructed with three fixed zeros, and replace the final normalization with the verified seven-scale, one-anchor parameterization.

EVIDENCE: The 1,532-parameter design achieved 99.94% accuracy after fixing one query-bias coordinate per head, while the one-anchor terminal scale previously achieved 100%; this extends the same headwise rotational gauge without narrowing query/key projections or removing terminal relative-scale capacity.

<<<<<<< SEARCH
class ScaleOnlyLayerNorm(nn.Module):
    """Layer normalization with a learned scale and no learned shift."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, (x.size(-1),), self.weight, None)
=======
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with all but one scale learned."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = F.pad(self.weight, (0, 1), value=1.0)
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(d_model))
        self.proj = MeanZeroLinear(d_model, d_model)
=======
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(n_head, 1))
        self.proj = MeanZeroLinear(d_model, d_model)
>>>>>>> REPLACE

<<<<<<< SEARCH
        q, k, v = qkv.chunk(3, dim=-1)
        q = q + self.q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
=======
        q, k, v = qkv.chunk(3, dim=-1)
        q_bias = F.pad(self.q_bias, (0, self.head_dim - 1)).reshape(d_model)
        q = q + q_bias

        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln_f = ScaleOnlyLayerNorm(cfg.d_model)
=======
        self.ln_f = AnchoredScaleLayerNorm(cfg.d_model)
>>>>>>> REPLACE
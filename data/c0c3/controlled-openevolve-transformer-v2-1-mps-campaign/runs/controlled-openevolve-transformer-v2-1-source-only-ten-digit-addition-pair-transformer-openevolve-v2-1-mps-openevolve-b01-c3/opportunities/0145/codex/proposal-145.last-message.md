MECHANISM: Endpoint-anchored normalization with singleton-versus-sextet attention bias

HYPOTHESIS: The resulting 853-parameter transformer will maintain at least 99% accuracy because the qualified 854-parameter endpoint-anchored LayerNorm design achieved 99.97%, and merging two adjacent attention output-bias groups removes one parameter without reducing embedding rank, MLP width, or attention-gain capacity.

INTENDED_EDIT: Adopt the qualified one-parameter final LayerNorm and reduce the attention output bias from `[a,b,b,c,c,c,c]` to `[a,b,b,b,b,b,b]`.

EVIDENCE: The endpoint-anchored 854-parameter model reached 99.97%, while prior failures came from eliminating embedding coefficients, reducing attention gains, shrinking the MLP, or removing the final-scale contrast; this tests a distinct isolated one-parameter tie.

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with one shared scale triple and one fixed anchor."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (self.weight, self.weight[-1:].expand(2), self.weight.new_ones(1))
        )
        return F.layer_norm(x, (x.size(-1),), weight, None)
=======
class AnchoredScaleLayerNorm(nn.Module):
    """Layer normalization with one shared scale sextet and fixed endpoints."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight.new_ones(1),
                self.weight.expand(x.size(-1) - 2),
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.output_bias = nn.Parameter(torch.zeros(d_model - 5))
=======
        self.output_bias = nn.Parameter(torch.zeros(d_model - 6))
>>>>>>> REPLACE

<<<<<<< SEARCH
        bias = torch.cat(
            (
                self.output_bias[:1],
                self.output_bias[1:2].expand(2),
                self.output_bias[-1:].expand(4),
            )
        )
=======
        bias = torch.cat(
            (
                self.output_bias[:1],
                self.output_bias[-1:].expand(self.output_dim - 1),
            )
        )
>>>>>>> REPLACE
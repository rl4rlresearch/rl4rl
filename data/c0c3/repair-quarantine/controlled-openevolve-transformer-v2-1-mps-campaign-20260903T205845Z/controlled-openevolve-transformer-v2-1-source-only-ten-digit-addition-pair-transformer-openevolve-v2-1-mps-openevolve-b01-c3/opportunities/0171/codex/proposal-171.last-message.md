MECHANISM: Two-pair final-normalization scale tying

HYPOTHESIS: The verified 746-parameter design will retain at least 99% accuracy when a second disjoint pair of final LayerNorm scales is tied, reducing the model to 745 parameters.

INTENDED_EDIT: Restore the verified four-parameter secondary-head gain pattern and expand five learned final-normalization scales as `[a,b,c,d,d,e,e,1]`.

EVIDENCE: Reference Design 3 achieved 99.8% accuracy with 746 parameters after one final-normalization scale tie, while further attention-gain compression failed at 90.67%; this tests another orthogonal normalization tie while preserving the verified attention pattern.

<<<<<<< SEARCH
class AnchoredScaleLayerNorm(nn.Module):
    """Bias-free LayerNorm with one fixed scale coordinate."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:-1],
                self.weight[-1:].expand(2),
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, (x.size(-1),), weight, None)
=======
class AnchoredScaleLayerNorm(nn.Module):
    """LayerNorm with two paired scales and one fixed anchor."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim - 3))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weight = torch.cat(
            (
                self.weight[:-2],
                self.weight[-2:].repeat_interleave(2),
                self.weight.new_ones(1),
            )
        )
        return F.layer_norm(x, (x.size(-1),), weight, None)
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 2)
        )
=======
        self.secondary_value_gain = nn.Parameter(
            torch.ones(n_head - 1, self.output_dim - 3)
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :3],
                self.secondary_value_gain[..., 3:].repeat_interleave(2, dim=-1),
            ),
            dim=-1,
        )
=======
        secondary_value_gain = torch.cat(
            (
                self.secondary_value_gain[..., :1],
                self.secondary_value_gain[..., 1:].repeat_interleave(2, dim=-1),
            ),
            dim=-1,
        )
>>>>>>> REPLACE
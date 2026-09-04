MECHANISM: Nonadjacent final-position residual-uniform gauge

HYPOTHESIS: Zero-centering positional rows 1–4 and the final positional row will yield 1,586 parameters with at least 99% accuracy, because it preserves the verified row-4 reduction while testing whether the row-5 failure was position-specific.

INTENDED_EDIT: Reproduce the qualified row-4 compaction and represent the nonadjacent final positional row in the same seven-dimensional zero-mean basis.

EVIDENCE: Centering rows 1–4 achieved 99.84% accuracy at 1,587 parameters, while extending contiguously through row 5 collapsed to 71.90%; the successful nonadjacent LayerNorm anchors show that avoiding an adjacent constrained coordinate can materially improve optimization.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and four row-shift gauges fixed."""
=======
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and six row-shift gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        centered_positions = embedding.weight[1:4] - embedding.weight[1:4].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[4:].detach().clone())
=======
        selected_positions = torch.cat(
            (embedding.weight[1:5], embedding.weight[-1:]),
            dim=0,
        )
        centered_positions = selected_positions - selected_positions.mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[5:-1].detach().clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        full_weight = torch.cat(
            (
                origin.unsqueeze(0),
                compact_positions,
                self.weight,
            ),
            dim=0,
        )
=======
        full_weight = torch.cat(
            (
                origin.unsqueeze(0),
                compact_positions[:-1],
                self.weight,
                compact_positions[-1:],
            ),
            dim=0,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix seven token/position translations, then center the singleton
        # positional-origin tail and positional rows 1 through 3.
=======
        # Fix seven token/position translations, then center the singleton
        # positional origin, rows 1 through 4, and the final positional row.
>>>>>>> REPLACE
MECHANISM: Fifth positional-row residual-uniform gauge quotient

HYPOTHESIS: Extending zero-centering from positional rows 1–4 through row 5 will reduce the verified 1,587-parameter model to 1,586 parameters while retaining at least 99% accuracy, because a position-specific uniform residual shift is removed by every pre-LayerNorm branch and the final LayerNorm.

INTENDED_EDIT: Represent positional row 5 in the existing seven-dimensional zero-mean basis and retain full parameters only from row 6 onward.

EVIDENCE: Zero-centering positional row 4 reduced the qualified 1,588-parameter design to 1,587 parameters and achieved 99.84% accuracy; row 5 has the same residual-uniform gauge structure, making it the most direct next reduction.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and five row-shift gauges fixed."""
=======
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and six row-shift gauges fixed."""
>>>>>>> REPLACE

<<<<<<< SEARCH
        centered_positions = embedding.weight[1:5] - embedding.weight[1:5].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[5:].detach().clone())
=======
        centered_positions = embedding.weight[1:6] - embedding.weight[1:6].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[6:].detach().clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix seven token/position translations, then center the singleton
        # positional-origin tail and positional rows 1 through 3.
=======
        # Fix seven token/position translations, then center the singleton
        # positional-origin tail and positional rows 1 through 5.
>>>>>>> REPLACE
MECHANISM: Combined positional LayerNorm-null gauge and pre-MLP bias absorption

HYPOTHESIS: Centering positional rows 1–6 and fixing a seventh `ln2` bias coordinate will yield 1,577 parameters with at least 99% accuracy, because each reduction independently exceeded 99.7% accuracy at 1,578 parameters.

INTENDED_EDIT: Extend positional centering from rows 1–4 to rows 1–6 and increase `ln2`’s fixed bias coordinates from six to seven.

EVIDENCE: Reference Design 3 centered positional rows 1–6 and achieved 99.73% accuracy at 1,578 parameters, while Reference Design 2 fixed seven `ln2` bias coordinates and achieved 99.97%; combining these independently qualified gauge reductions is the closest supported path to 1,577 parameters.

<<<<<<< SEARCH
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and five row-shift gauges fixed."""
=======
class CompactPositionEmbedding(nn.Module):
    """Position embedding with seven translations and seven row-shift gauges fixed."""
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
        centered_positions = embedding.weight[1:7] - embedding.weight[1:7].mean(
            dim=1,
            keepdim=True,
        )
        self.position_weight = nn.Parameter(
            (centered_positions @ position_basis).detach().clone()
        )
        self.weight = nn.Parameter(embedding.weight[7:].detach().clone())
>>>>>>> REPLACE

<<<<<<< SEARCH
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=6,
            fixed_weight_coordinates=2,
        )
=======
        self.ln2 = CompactLayerNormBias(
            nn.LayerNorm(cfg.d_model),
            fixed_coordinates=7,
            fixed_weight_coordinates=2,
        )
>>>>>>> REPLACE

<<<<<<< SEARCH
        # Fix seven token/position translations, then center the singleton
        # positional-origin tail and positional rows 1 through 4.
=======
        # Fix seven token/position translations, then center the singleton
        # positional-origin tail and positional rows 1 through 6.
>>>>>>> REPLACE
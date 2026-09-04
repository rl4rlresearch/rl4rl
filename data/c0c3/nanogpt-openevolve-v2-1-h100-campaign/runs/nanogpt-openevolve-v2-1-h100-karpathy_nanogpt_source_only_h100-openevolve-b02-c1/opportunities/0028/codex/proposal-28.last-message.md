MECHANISM: Fine-grained cautious Muon decay reduction

HYPOTHESIS: Reducing Muon weight decay from 0.14 to 0.12 will preserve the benefit of update-count normalization while avoiding residual over-regularization, lowering val_bpb below 0.984418.

INTENDED_EDIT: Lower only the cautious Muon weight-decay coefficient, preserving the best 176K-token batch regime and all other settings.

EVIDENCE: Scaling weight decay from 0.20 to 0.14 improved val_bpb from 0.986967 to 0.984418, whereas increasing Muon beta2 regressed to 0.984760; a smaller decay adjustment is the most direct refinement of the successful mechanism.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.14     # preserve cumulative decay as update count increases
=======
WEIGHT_DECAY = 0.12     # refine update-count-normalized cautious decay
>>>>>>> REPLACE
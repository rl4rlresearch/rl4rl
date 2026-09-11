MECHANISM: Interpolated cautious-decay optimum

HYPOTHESIS: Setting Muon weight decay to 0.125 will lower val_bpb below 0.985746 by retaining more regularization than the unsuccessful 0.05 setting while avoiding the over-regularization observed at 0.2.

INTENDED_EDIT: Keep the proven 262K-token, 128-sequence design unchanged and set cautious Muon weight decay to 0.125.

EVIDENCE: Weight decay 0.1 achieved the best val_bpb of 0.985746, while both 0.2 (0.987182) and 0.05 (0.987328) were worse; interpolation of these results motivates testing a slightly stronger value near the apparent interior optimum.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
=======
WEIGHT_DECAY = 0.125    # interpolate near the observed cautious-decay optimum
>>>>>>> REPLACE
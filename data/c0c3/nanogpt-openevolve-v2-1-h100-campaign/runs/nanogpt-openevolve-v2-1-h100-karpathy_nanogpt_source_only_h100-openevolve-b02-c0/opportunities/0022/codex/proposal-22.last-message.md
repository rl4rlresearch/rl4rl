MECHANISM: Quadratic-decay strength retuning

HYPOTHESIS: Increasing initial Muon weight decay from 0.1 to 0.125 while retaining quadratic annealing will reduce val_bpb below 0.986327 by adding useful early regularization without restoring excessive late-training shrinkage.

INTENDED_EDIT: Raise the initial Muon weight-decay coefficient to 0.125; preserve the quadratic schedule and all other settings.

EVIDENCE: Quadratic annealing at 0.1 produced the best val_bpb, 0.986327, while further exponent refinements from 1.75 through 3.0 did not improve it. Since quadratic annealing has lower cumulative decay than the previously tested linear schedule, retuning its initial magnitude is the most informative remaining test on the successful decay axis.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.1      # preserve cumulative decay horizon after doubling optimizer updates
=======
WEIGHT_DECAY = 0.125    # strengthen early regularization under quadratic decay
>>>>>>> REPLACE
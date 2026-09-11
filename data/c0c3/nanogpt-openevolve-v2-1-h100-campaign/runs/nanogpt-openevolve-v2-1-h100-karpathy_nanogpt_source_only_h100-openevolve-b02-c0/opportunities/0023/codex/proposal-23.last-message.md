MECHANISM: Quadratic-decay strength continuation

HYPOTHESIS: Increasing initial Muon weight decay from 0.125 to 0.15 while retaining quadratic annealing will reduce val_bpb below 0.986290 by extending the improvement observed when increasing it from 0.1 to 0.125.

INTENDED_EDIT: Raise the initial Muon weight-decay coefficient to 0.15 while preserving its quadratic schedule and all other settings.

EVIDENCE: Quadratic decay at 0.125 achieved 0.986290 versus 0.986327 at 0.1, while decay-exponent refinements failed to improve the result; testing a higher coefficient is the most direct continuation of the currently successful axis.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.125    # strengthen early regularization under quadratic decay
=======
WEIGHT_DECAY = 0.15     # continue tuning early regularization under quadratic decay
>>>>>>> REPLACE
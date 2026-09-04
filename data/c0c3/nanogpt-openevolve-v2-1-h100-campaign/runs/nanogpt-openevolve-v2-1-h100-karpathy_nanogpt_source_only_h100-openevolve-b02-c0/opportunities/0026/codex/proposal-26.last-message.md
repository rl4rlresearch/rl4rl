MECHANISM: Bracketed quadratic-decay strength interpolation

HYPOTHESIS: Setting initial Muon weight decay to 0.15625 will reduce val_bpb below 0.985730 by refining the optimum bracketed by the winning 0.15 and regressing 0.175 settings.

INTENDED_EDIT: Increase initial Muon weight decay from 0.15 to 0.15625 while preserving quadratic annealing and all other settings.

EVIDENCE: Quadratic-decayed weight decay achieved 0.985730 at 0.15, while 0.175 and 0.20 regressed to 0.985999 and 0.985889; a smaller step above 0.15 tests whether the optimum lies just inside that bracket.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.15     # continue tuning early regularization under quadratic decay
=======
WEIGHT_DECAY = 0.15625  # refine the bracketed optimum under quadratic decay
>>>>>>> REPLACE
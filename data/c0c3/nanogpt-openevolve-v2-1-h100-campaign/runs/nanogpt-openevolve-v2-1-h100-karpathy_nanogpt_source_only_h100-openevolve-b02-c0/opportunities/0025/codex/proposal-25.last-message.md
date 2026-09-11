MECHANISM: Bracketed quadratic-decay strength refinement

HYPOTHESIS: Increasing initial Muon weight decay from 0.15 to 0.175 while retaining quadratic annealing will reduce val_bpb below 0.985730 by refining the bracket between the best 0.15 result and the slightly worse 0.20 result.

INTENDED_EDIT: Set the initial Muon weight-decay coefficient to 0.175, leaving its quadratic schedule and all other settings unchanged.

EVIDENCE: Quadratic-decayed weight decay improved val_bpb from 0.986290 at 0.125 to 0.985730 at 0.15, but 0.20 regressed slightly to 0.985889; testing their midpoint is the most direct refinement of the newly bracketed optimum.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.15     # continue tuning early regularization under quadratic decay
=======
WEIGHT_DECAY = 0.175    # refine bracketed optimum under quadratic decay
>>>>>>> REPLACE
MECHANISM: Bracketed quadratic-decay strength interpolation

HYPOTHESIS: Reducing initial Muon weight decay from 0.15 to 0.14375 will improve val_bpb below 0.985730 by testing the unexplored side of the local optimum after 0.15625 regressed.

INTENDED_EDIT: Set initial Muon weight decay to 0.14375 while preserving quadratic annealing and all other settings.

EVIDENCE: Quadratic decay achieved the best result at 0.15, while the nearby increase to 0.15625 worsened val_bpb from 0.985730 to 0.985961; an equal-sized step below 0.15 provides the most direct local refinement.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.15     # continue tuning early regularization under quadratic decay
=======
WEIGHT_DECAY = 0.14375  # refine the lower side of the quadratic-decay optimum
>>>>>>> REPLACE
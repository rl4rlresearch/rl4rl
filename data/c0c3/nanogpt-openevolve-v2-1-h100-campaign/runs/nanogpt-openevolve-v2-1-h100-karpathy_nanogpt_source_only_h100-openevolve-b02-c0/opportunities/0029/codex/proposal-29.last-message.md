MECHANISM: Earlier terminal learning-rate annealing

HYPOTHESIS: Increasing the warmdown fraction from 0.5 to 0.6 will reduce val_bpb below 0.985730 by beginning learning-rate decay earlier and allowing more stable late-stage convergence.

INTENDED_EDIT: Retain the best quadratic Muon weight decay of 0.15 and extend linear learning-rate warmdown to the final 60% of training.

EVIDENCE: Shortening warmdown from 0.5 to 0.4 slightly regressed val_bpb from 0.985730 to 0.985770, motivating a symmetric test on the longer-warmdown side while leaving the locally optimized weight decay unchanged.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.6    # begin terminal LR annealing earlier for a longer convergence phase
>>>>>>> REPLACE
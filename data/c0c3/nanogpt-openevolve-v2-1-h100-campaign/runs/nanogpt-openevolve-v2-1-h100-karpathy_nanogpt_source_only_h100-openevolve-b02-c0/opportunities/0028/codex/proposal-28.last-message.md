MECHANISM: Shorter terminal learning-rate warmdown

HYPOTHESIS: Reducing the warmdown fraction from 0.5 to 0.4 will lower val_bpb below 0.985730 by preserving productive learning rates longer while still annealing them to zero before validation.

INTENDED_EDIT: Retain the best quadratic Muon weight decay of 0.15 and shorten the linear learning-rate warmdown to the final 40% of training.

EVIDENCE: Weight decay 0.15 achieved the best val_bpb, while nearby coefficients 0.14375, 0.15625, 0.175, and 0.20 all regressed; this locally brackets that axis and motivates testing an independent schedule parameter without disturbing the winning regularization.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.4    # retain peak learning rates longer before final annealing
>>>>>>> REPLACE
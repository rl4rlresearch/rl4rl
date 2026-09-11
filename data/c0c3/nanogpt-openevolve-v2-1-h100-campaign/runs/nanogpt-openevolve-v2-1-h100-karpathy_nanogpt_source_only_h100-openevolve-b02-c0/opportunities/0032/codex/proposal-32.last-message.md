MECHANISM: Bracketed linear warmdown-duration refinement

HYPOTHESIS: A 52.5% linear warmdown will reduce val_bpb below 0.985730 by slightly extending late-stage annealing without incurring the substantial regression observed at 60%.

INTENDED_EDIT: Increase WARMDOWN_RATIO from 0.50 to 0.525 while preserving the best quadratic Muon weight decay and linear learning-rate schedule.

EVIDENCE: The 50% warmdown achieved 0.985730, while 45% regressed slightly to 0.985842 and 60% regressed sharply to 0.986996; testing 52.5% refines the unexplored side of the bracket around the current optimum.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.525  # refine the bracket between the best 50% and regressing 60% warmdowns
>>>>>>> REPLACE
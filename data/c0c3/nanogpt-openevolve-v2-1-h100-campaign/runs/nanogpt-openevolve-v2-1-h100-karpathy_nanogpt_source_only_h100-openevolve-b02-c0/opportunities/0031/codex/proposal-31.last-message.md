MECHANISM: Bracketed linear warmdown-duration interpolation

HYPOTHESIS: A 45% linear warmdown will reduce val_bpb below 0.985730 by retaining productive learning rates slightly longer than the best 50% schedule without moving as far as the nearly tied 40% schedule.

INTENDED_EDIT: Set WARMDOWN_RATIO to 0.45 while preserving the best Muon weight decay and linear schedule.

EVIDENCE: The 40% warmdown nearly matched the 50% optimum (0.985770 versus 0.985730), while 60% regressed sharply to 0.986996; interpolating at 45% is the most informative refinement of this bracket.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.45   # refine the bracket between the near-tied 40% and 50% schedules
>>>>>>> REPLACE
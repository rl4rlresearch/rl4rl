MECHANISM: Earlier linear learning-rate annealing

HYPOTHESIS: Extending linear warmdown from 50% to 60% of training will reduce val_bpb below 0.985757 by avoiding the larger early-annealing updates implicated by the worse cosine result.

INTENDED_EDIT: Preserve the winning 112-sequence batch and optimizer settings, but begin the linear learning-rate decay at 40% rather than 50% of the training window.

EVIDENCE: Cosine warmdown regressed val_bpb from 0.985757 to 0.988710 while retaining larger learning rates early in annealing, motivating an earlier reduction using the proven linear schedule.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.6    # begin linear LR decay after 40% of the time budget
>>>>>>> REPLACE
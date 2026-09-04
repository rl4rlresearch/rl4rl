MECHANISM: Earlier linear learning-rate annealing

HYPOTHESIS: Extending linear warmdown from 50% to 60% will beat 0.985318 val_bpb by shifting more of the fixed window toward progressively smaller refinement updates.

INTENDED_EDIT: Preserve the best architecture, batch size, and linear schedule while starting warmdown 10% earlier.

EVIDENCE: Shortening linear warmdown to 40% regressed val_bpb from 0.985318 to 0.986825, directly motivating the adjacent test in the opposite direction.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.6    # fraction of time budget for LR warmdown
>>>>>>> REPLACE
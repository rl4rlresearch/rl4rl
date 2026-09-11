MECHANISM: Earlier linear learning-rate annealing

HYPOTHESIS: Extending linear warmdown from 70% to 80% will beat 0.985071 val_bpb by allocating more of the fixed training window to progressively smaller refinement updates.

INTENDED_EDIT: Preserve the best architecture, batch size, and linear schedule while starting warmdown 10% earlier.

EVIDENCE: Linear warmdown improved monotonically from 0.985318 at 50% to 0.985215 at 60% and 0.985071 at 70%, motivating the next adjacent test in the successful direction.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.7    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.8    # fraction of time budget for LR warmdown
>>>>>>> REPLACE
MECHANISM: Earlier linear learning-rate annealing

HYPOTHESIS: Extending linear warmdown from 60% to 70% will beat 0.985215 val_bpb by allocating more of the fixed window to progressively smaller refinement updates.

INTENDED_EDIT: Preserve the best architecture, batch size, and linear schedule while starting warmdown 10% earlier.

EVIDENCE: Moving from 50% to 60% warmdown improved val_bpb from 0.985318 to 0.985215, while shortening it to 40% regressed to 0.986825, motivating the next step in the successful direction.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.6    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.7    # fraction of time budget for LR warmdown
>>>>>>> REPLACE
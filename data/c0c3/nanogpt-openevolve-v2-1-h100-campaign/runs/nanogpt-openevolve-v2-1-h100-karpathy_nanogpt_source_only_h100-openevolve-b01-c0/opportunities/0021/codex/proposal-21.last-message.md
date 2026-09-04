MECHANISM: Updated quadratic-interpolated linear warmdown

HYPOTHESIS: A 75.6% linear warmdown will beat 0.984455 val_bpb by moving to the revised local minimum estimated from the 70%, 78.5%, and 80% results.

INTENDED_EDIT: Reduce `WARMDOWN_RATIO` from 0.785 to 0.756, delaying the start of the existing linear decay while preserving all other settings.

EVIDENCE: The 78.5% run improved val_bpb to 0.984455 versus 0.985071 at 70% and 0.984745 at 80%; quadratic interpolation of these nearest observations estimates the minimum near 75.6%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.785  # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.756  # fraction of time budget for LR warmdown
>>>>>>> REPLACE
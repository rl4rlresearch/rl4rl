MECHANISM: Earlier linear learning-rate decay

HYPOTHESIS: Extending linear warmdown from 62.5% to 75% of training will lower val_bpb below 0.983689 by reducing mid-training update magnitudes while preserving the proven linear decay shape.

INTENDED_EDIT: Keep the best SMMLSSSL architecture and all optimizer magnitudes fixed, but begin linear warmdown at 25% rather than 37.5% of the time budget.

EVIDENCE: Cosine warmdown regressed from 0.983689 to 0.986693 while sustaining higher learning rates during early warmdown; an earlier linear decay directly tests whether smaller mid-training updates improve convergence without changing throughput or architecture.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.625  # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.75   # begin linear decay earlier to reduce mid-training update magnitudes
>>>>>>> REPLACE
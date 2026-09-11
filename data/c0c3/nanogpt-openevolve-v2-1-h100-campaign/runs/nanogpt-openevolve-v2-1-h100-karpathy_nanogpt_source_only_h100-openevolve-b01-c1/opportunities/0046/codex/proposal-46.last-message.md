MECHANISM: Earlier linear learning-rate decay

HYPOTHESIS: Starting linear warmdown at 25% of training will lower val_bpb below 0.983689 by reducing mid-training update magnitudes while retaining the proven linear decay shape.

INTENDED_EDIT: Increase WARMDOWN_RATIO from 0.625 to 0.75, leaving the best SMMLSSSL architecture and all optimizer magnitudes unchanged.

EVIDENCE: Cosine warmdown regressed to 0.986693 while sustaining larger mid-training learning rates; the previously proposed earlier linear decay was not verified, so implementing it cleanly is the most direct follow-up.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.625  # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.75   # begin linear warmdown at 25% of the time budget
>>>>>>> REPLACE
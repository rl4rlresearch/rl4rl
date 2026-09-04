MECHANISM: Later linear learning-rate warmdown

HYPOTHESIS: Starting linear warmdown at 50% rather than 37.5% of training will lower val_bpb below 0.983689 by preserving full-size productive updates longer while retaining the proven linear taper to zero.

INTENDED_EDIT: Reduce WARMDOWN_RATIO from 0.625 to 0.5, leaving the best SMMLSSSL architecture and all optimizer magnitudes unchanged.

EVIDENCE: Moving warmdown earlier by increasing WARMDOWN_RATIO to 0.75 regressed val_bpb from 0.983689 to 0.984253; testing the opposite direction is the most informative adjacent schedule change.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.625  # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
>>>>>>> REPLACE
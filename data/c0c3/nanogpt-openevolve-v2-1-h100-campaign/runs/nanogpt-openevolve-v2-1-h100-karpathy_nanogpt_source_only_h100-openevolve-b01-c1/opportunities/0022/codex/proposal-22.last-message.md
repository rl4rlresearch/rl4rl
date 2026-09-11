MECHANISM: Extended linear learning-rate warmdown

HYPOTHESIS: Starting warmdown at 30% rather than 40% of the training window will further reduce late-stage optimization noise at the successful 262K-token batch size and lower val_bpb below 0.985493.

INTENDED_EDIT: Increase WARMDOWN_RATIO from 0.6 to 0.7 while preserving architecture, batch size, peak learning rates, throughput, and zero final learning rate.

EVIDENCE: Increasing WARMDOWN_RATIO from 0.5 to 0.6 improved val_bpb from 0.986167 to 0.985493 at effectively unchanged tokens and steps, directly motivating a continuation of the warmdown-duration sweep.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.6    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.7    # fraction of time budget for LR warmdown
>>>>>>> REPLACE
MECHANISM: Bracketed linear warmdown refinement

HYPOTHESIS: Starting linear warmdown at 35% of training will retain roughly 528M tokens and lower val_bpb below 0.992358.

INTENDED_EDIT: Extend linear LR warmdown from 50% to 65% of the training window while preserving the proven 256-token attention architecture, optimizer, and batching.

EVIDENCE: A 60% linear warmdown achieved the best observed val_bpb of 0.992358, improving over 50% at 0.992854, while 70% regressed to 0.993193; 65% directly probes the bracketed schedule optimum.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.65   # begin linear LR warmdown after 35% of the time budget
>>>>>>> REPLACE
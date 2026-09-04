MECHANISM: Quadratic-guided warmdown refinement

HYPOTHESIS: Setting WARMDOWN_RATIO to 0.625 will better balance sustained peak learning with late-stage noise reduction and lower val_bpb below 0.985493.

INTENDED_EDIT: Start linear learning-rate warmdown at 37.5% of the training window while preserving architecture, batch size, peak learning rates, and final zero learning rate.

EVIDENCE: WARMDOWN_RATIO 0.6 improved val_bpb from 0.986167 to 0.985493, while 0.7 slightly regressed to 0.985706; the asymmetric three-point sweep suggests the local optimum lies just above 0.6.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.6    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.625  # fraction of time budget for LR warmdown
>>>>>>> REPLACE
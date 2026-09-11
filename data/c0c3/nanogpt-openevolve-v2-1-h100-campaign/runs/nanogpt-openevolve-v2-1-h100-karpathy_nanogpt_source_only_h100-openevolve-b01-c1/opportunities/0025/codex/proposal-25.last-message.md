MECHANISM: Quadratic-interpolated warmdown timing

HYPOTHESIS: Setting WARMDOWN_RATIO to 0.645 will lower val_bpb below 0.985229 by placing warmdown near the local optimum implied by the 0.6, 0.625, and 0.7 results.

INTENDED_EDIT: Increase WARMDOWN_RATIO from 0.625 to 0.645 while preserving architecture, batching, peak learning rates, and final zero learning rate.

EVIDENCE: WARMDOWN_RATIO 0.625 achieved the best val_bpb of 0.985229, improving on 0.6 at 0.985493, while 0.7 regressed to 0.985706; quadratic interpolation of these three controlled results places the estimated minimum near 0.644.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.625  # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.645  # fraction of time budget for LR warmdown
>>>>>>> REPLACE
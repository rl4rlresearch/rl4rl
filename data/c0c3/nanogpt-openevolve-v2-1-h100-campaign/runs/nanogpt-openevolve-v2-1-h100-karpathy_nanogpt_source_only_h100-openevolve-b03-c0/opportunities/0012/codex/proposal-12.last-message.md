MECHANISM: Evidence-guided warmdown interpolation

HYPOTHESIS: Setting `WARMDOWN_RATIO` to `0.54` will achieve `val_bpb < 0.986636` by placing the decay onset near the quadratic optimum implied by the tested 0.4, 0.5, and 0.6 schedules.

INTENDED_EDIT: Increase warmdown from 50% to 54% of training while preserving architecture, batching, learning rates, and throughput.

EVIDENCE: Warmdown ratios 0.4, 0.5, and 0.6 produced `val_bpb` values of 0.988601, 0.986636, and 0.986873 respectively; the asymmetric regressions around 0.5 imply a local optimum near 0.54.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.54   # fraction of time budget for LR warmdown
>>>>>>> REPLACE
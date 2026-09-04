MECHANISM: Quadratic refinement of depth-biased local context

HYPOTHESIS: Compute-matched 140-token early and 132-token late windows will lower val_bpb below 0.983275 by placing the depth bias near the quadratic minimum implied by the uniform, moderate-bias, and strong-bias results.

INTENDED_EDIT: Reduce the early/late local-window contrast from 144/128 to 140/132 while preserving the 136-token average, full-context layers 4 and 8, and all optimizer settings.

EVIDENCE: Uniform 136/136 achieved 0.983317, moderate 144/128 improved to 0.983275, and stronger 152/120 regressed to 0.983933; quadratic interpolation places the estimated optimum near 140.5/131.5.

<<<<<<< SEARCH
        early_short_window = 9 * long_window // 128
        late_short_window = long_window // 16
=======
        early_short_window = 35 * long_window // 512
        late_short_window = 33 * long_window // 512
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # 144-token early and 128-token late local layers; full context at layers 4 and 8
=======
WINDOW_PATTERN = "SSSL" # 140-token early and 132-token late local layers; full context at layers 4 and 8
>>>>>>> REPLACE
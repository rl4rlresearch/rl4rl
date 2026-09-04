MECHANISM: Compute-neutral moderate front-loading of local context

HYPOTHESIS: Using 448-token early windows and 320-token late windows will reduce `val_bpb` below 0.983618 by favoring early context without starving late refinement or changing total local-attention compute.

INTENDED_EDIT: Restore an average 384-token local window while reallocating 64 tokens from each late local layer to its early counterpart; retain full context at layers 4 and 8.

EVIDENCE: Uniform 384-token attention achieved 0.983618, and the front-loaded 512/256 split nearly matched it at 0.983620 while the reverse 256/512 split regressed to 0.984213; the midpoint tests the indicated direction with less extreme late-context reduction.

<<<<<<< SEARCH
        early_short_window = long_window // 8
        late_short_window = long_window // 4
=======
        early_short_window = 7 * long_window // 32
        late_short_window = 5 * long_window // 32
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows widen from 256 to 512
=======
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 448 to 320
>>>>>>> REPLACE
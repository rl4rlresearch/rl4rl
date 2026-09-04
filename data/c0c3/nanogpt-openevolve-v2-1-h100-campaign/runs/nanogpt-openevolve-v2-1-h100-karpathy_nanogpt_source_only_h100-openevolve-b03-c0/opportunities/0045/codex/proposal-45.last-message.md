MECHANISM: Quadratically interpolated late-global context allocation

HYPOTHESIS: A 1,152-token final-layer window will achieve `val_bpb < 0.981559` by balancing the degradation observed at 896 and 1,280 tokens around the 1,024-token optimum.

INTENDED_EDIT: Increase only the final attention window from 1,024 to 1,152 tokens, preserving all other settings.

EVIDENCE: Final-window results of `0.982366` at 896, `0.981559` at 1,024, and `0.981659` at 1,280 place a quadratic interpolation minimum near 1,141 tokens, motivating the aligned 1,152-token test.

<<<<<<< SEARCH
        window_sizes[-1] = (long_window // 2, 0)
=======
        window_sizes[-1] = (9 * long_window // 16, 0)
>>>>>>> REPLACE
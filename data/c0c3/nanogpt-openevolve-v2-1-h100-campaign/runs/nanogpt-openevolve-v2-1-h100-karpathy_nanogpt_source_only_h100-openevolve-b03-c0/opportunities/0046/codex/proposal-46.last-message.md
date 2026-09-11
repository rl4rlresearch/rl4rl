MECHANISM: Interpolated late-global context allocation

HYPOTHESIS: A 1,152-token final-layer window will achieve `val_bpb < 0.981559`, balancing the degradation observed on either side of the 1,024-token result.

INTENDED_EDIT: Increase only the final attention window from 1,024 to 1,152 tokens while preserving every other setting.

EVIDENCE: Final-window results were `0.982366` at 896 tokens, `0.981559` at 1,024, and `0.981659` at 1,280; quadratic interpolation places the estimated minimum near 1,141 tokens, making the 128-aligned 1,152-token window the most targeted unresolved test.

<<<<<<< SEARCH
        window_sizes[-1] = (long_window // 2, 0)
=======
        window_sizes[-1] = (long_window * 9 // 16, 0)
>>>>>>> REPLACE
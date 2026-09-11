MECHANISM: Quadratically refined early local-to-global context allocation

HYPOTHESIS: A 480-token first pre-global window will achieve `val_bpb < 0.981072` by retaining the benefit of expanded early context while avoiding the over-allocation observed at 1,024 tokens.

INTENDED_EDIT: Reduce only the first pre-global attention window from 512 to 480 tokens; preserve the 1,152-token final window and every other setting.

EVIDENCE: First pre-global windows of 256, 512, and 1,024 tokens yielded `0.982234`, `0.982068`, and `0.983064`; quadratic interpolation places the estimated optimum near 480 tokens.

<<<<<<< SEARCH
        pre_global_window = long_window // 4
=======
        pre_global_window = long_window * 15 // 64
>>>>>>> REPLACE
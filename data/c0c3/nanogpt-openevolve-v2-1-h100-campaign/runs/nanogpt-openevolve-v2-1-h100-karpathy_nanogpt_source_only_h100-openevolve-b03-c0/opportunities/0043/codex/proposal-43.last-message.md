MECHANISM: Interpolated late-global context allocation

HYPOTHESIS: A 1,280-token final-layer window will achieve `val_bpb < 0.981559` by retaining the regularization benefit of reduced late attention while restoring context lost by the unsuccessful 512-token window.

INTENDED_EDIT: Increase only the final attention window from 1,024 to 1,280 tokens; preserve all other architecture, optimizer, batching, and schedule settings.

EVIDENCE: The 1,024-token final window improved `val_bpb` from `0.982068` to `0.981559`, while 512 tokens regressed sharply to `0.985963`; testing 1,280 tokens probes the promising interval between the best reduced window and full 2,048-token attention.

<<<<<<< SEARCH
        window_sizes[-1] = (long_window // 2, 0)
=======
        window_sizes[-1] = (5 * long_window // 8, 0)
>>>>>>> REPLACE
MECHANISM: Progressive late-global context compression

HYPOTHESIS: Halving the final layer’s attention window from 1,024 to 512 tokens will achieve `val_bpb < 0.981559` by reallocating late long-range attention compute while the layer-4 full-context attention preserves global information.

INTENDED_EDIT: Reduce only the final attention layer’s window to one-quarter of the 2,048-token sequence; preserve all other architecture, optimizer, batching, and schedule settings.

EVIDENCE: Reducing only the final layer from 2,048-token full attention to 1,024 tokens improved `val_bpb` from `0.982068` to `0.981559`, the strongest result observed, directly motivating one further power-of-two reduction.

<<<<<<< SEARCH
        window_sizes[-1] = (long_window // 2, 0)
=======
        window_sizes[-1] = (long_window // 4, 0)
>>>>>>> REPLACE
MECHANISM: Expanded local-to-global boundary context

HYPOTHESIS: Using 512-token windows only for the two local layers immediately before full-context layers will achieve `val_bpb < 0.982545` by strengthening high-value aggregation while retaining 128-token windows in the other four local layers.

INTENDED_EDIT: Increase the pre-global attention window from 256 to 512 tokens, leaving all other architecture, optimizer, batching, and schedule settings unchanged.

EVIDENCE: The mixed 128/256 design improved `val_bpb` from `0.982806` for uniform 128-token windows to `0.982545`, despite slightly lower throughput, indicating that extra context is especially valuable immediately before global attention; the 512-token uniform design also confirms this window size is efficient and trainable.

<<<<<<< SEARCH
        pre_global_window = long_window // 8
=======
        pre_global_window = long_window // 4
>>>>>>> REPLACE
MECHANISM: Front-loaded global-context compute reallocation

HYPOTHESIS: Halving only the final layer’s attention window to 1,024 tokens will increase throughput enough to achieve `val_bpb < 0.982068`, while the earlier full-context layer preserves long-range information.

INTENDED_EDIT: Retain the 512-token first pre-global window and all 128-token local windows, but reduce the final attention layer from 2,048-token full context to a power-of-two 1,024-token window.

EVIDENCE: The best result (`0.982068`) concentrates extra local context before the first global layer, while adding context at the later pre-global boundary worsened `val_bpb` from `0.982234` to `0.982545`; this motivates testing whether late global-attention compute is also less valuable.

<<<<<<< SEARCH
        window_sizes[-1] = (long_window, 0)
=======
        window_sizes[-1] = (long_window // 2, 0)
>>>>>>> REPLACE
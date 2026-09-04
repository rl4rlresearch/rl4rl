MECHANISM: Further short-window context-compute reallocation

HYPOTHESIS: Reducing the six local-attention layers from 512-token to 256-token windows while retaining two full-context layers will increase token throughput enough to achieve `val_bpb < 0.983023`.

INTENDED_EDIT: Use one-eighth-context windows for `S` layers without changing model capacity, optimizer, batching, schedule, softcap, or full-context layers.

EVIDENCE: Reducing local windows from 1,024 to 512 tokens increased throughput from 493.2M to 509.1M tokens and improved `val_bpb` from `0.984431` to `0.983023`; this continuation tests whether further compute reallocation remains beneficial.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = long_window // 8
>>>>>>> REPLACE
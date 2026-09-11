MECHANISM: Short-window context-compute reallocation

HYPOTHESIS: Reducing short-attention windows from 1,024 to 512 tokens while retaining two full-context layers will increase token throughput enough to achieve `val_bpb < 0.984431`.

INTENDED_EDIT: Use quarter-context windows for `S` layers without changing model capacity, batching, optimizer, schedule, or full-context layers.

EVIDENCE: The best design processed 493.2M tokens, while recent slower designs processing 474.8M–489.4M tokens produced worse `val_bpb`; shortening only the six local-attention layers targets additional training tokens while preserving full-context attention in layers 4 and 8.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = long_window // 4
>>>>>>> REPLACE
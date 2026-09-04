MECHANISM: Throughput-biased three-eighths sliding attention

HYPOTHESIS: Reducing short-layer context from 1,024 to 768 tokens while retaining two full-context layers will increase token throughput enough to lower val_bpb below 0.984068.

INTENDED_EDIT: Set each `S` attention window to three-eighths of the 2,048-token sequence length.

EVIDENCE: The best design processed 478.5M tokens at 0.984068, while the slower delay-line design processed 442.1M and regressed to 0.986236; shortening only the six local-attention windows targets throughput without removing full-context attention.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = 3 * long_window // 8
>>>>>>> REPLACE
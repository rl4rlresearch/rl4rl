MECHANISM: Intermediate 768-token sliding attention

HYPOTHESIS: A 768-token short window will beat 0.993324 val_bpb by recovering some long-range modeling capacity while retaining most of the throughput advantage of 512-token windows.

INTENDED_EDIT: Set S layers to three-eighths of the 2048-token context while preserving the SSSL pattern and full-attention layers 4 and 8.

EVIDENCE: The 512-token design outperformed 1024 tokens, but reducing further to 256 regressed; testing the midpoint between 512 and 1024 probes whether the fixed-time optimum lies just above 512.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = 3 * long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=three-eighths context
>>>>>>> REPLACE
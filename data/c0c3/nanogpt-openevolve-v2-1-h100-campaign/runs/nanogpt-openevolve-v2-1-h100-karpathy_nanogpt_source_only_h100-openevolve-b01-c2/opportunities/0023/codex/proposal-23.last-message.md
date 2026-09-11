MECHANISM: Bracketed 224-token local-attention refinement

HYPOTHESIS: Seven 224-token local layers followed by the full-context final anchor will retain at least 525M training tokens while improving val_bpb below 0.992854.

INTENDED_EDIT: Reduce the current seven 512-token local-attention windows to 224 tokens, preserving the architecture, optimizer, batch size, schedule, and final global anchor.

EVIDENCE: The 256-token design achieved the best observed val_bpb of 0.992854, while 192 tokens regressed to 0.993246 despite higher throughput; 224 tokens directly probes the bracketed optimum between them.

<<<<<<< SEARCH
        long_window = config.sequence_len
        short_window = long_window // 4
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
=======
        long_window = config.sequence_len
        short_window = 7 * long_window // 64
        char_to_window = {"L": (long_window, 0), "S": (short_window, 0)}
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSS" # seven quarter-context layers, then one forced full-context anchor
=======
WINDOW_PATTERN = "SSSS" # seven 224-token local layers, then one forced full-context anchor
>>>>>>> REPLACE
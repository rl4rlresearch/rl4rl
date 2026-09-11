MECHANISM: Intermediate local-attention window interpolation

HYPOTHESIS: A 768-token short window will retain more context than 512 while processing more than 497M tokens, reducing val_bpb below 0.993287.

INTENDED_EDIT: Set the six short-attention layers to three-eighths of the 2048-token context.

EVIDENCE: Moving from 1024 to 512 tokens improved val_bpb from 0.995558 to 0.993287, while 256 tokens regressed to 1.009565 with a throughput cliff; testing the midpoint isolates the favorable window range.

<<<<<<< SEARCH
        short_window = long_window // 2
=======
        short_window = 3 * long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=three-eighths context
>>>>>>> REPLACE
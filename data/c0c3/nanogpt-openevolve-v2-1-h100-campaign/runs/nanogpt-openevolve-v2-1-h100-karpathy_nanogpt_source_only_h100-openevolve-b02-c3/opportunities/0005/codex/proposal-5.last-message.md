MECHANISM: Higher-frequency optimizer updates at the proven attention geometry

HYPOTHESIS: Halving the optimizer batch to 262K tokens while restoring the best 512-token SSSL attention design will nearly double update frequency with only modest throughput loss, reducing val_bpb below 0.993287.

INTENDED_EDIT: Restore quarter-context short attention and reduce gradient accumulation from two microbatches to one by halving TOTAL_BATCH_SIZE.

EVIDENCE: The 512-token SSSL design achieved the best result, 0.993287 val_bpb at 513.3M tokens, but made only 979 optimizer updates; the 768-token and 256-token window changes both caused throughput cliffs and worse validation, motivating an optimization-frequency test at the proven window size.

<<<<<<< SEARCH
        short_window = 3 * long_window // 8
=======
        short_window = long_window // 4
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=three-eighths context

# Optimization
TOTAL_BATCH_SIZE = 2**19 # ~524K tokens per optimizer step
=======
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=quarter context

# Optimization
TOTAL_BATCH_SIZE = 2**18 # ~262K tokens per optimizer step
>>>>>>> REPLACE
MECHANISM: Width-preserving depth reduction

HYPOTHESIS: Reducing depth from 8 to 7 while retaining the same 512-wide hidden dimension will increase token throughput by roughly 10% and achieve `val_bpb < 0.986636`.

INTENDED_EDIT: Set `DEPTH` to 7; dimension rounding preserves the current width, while one fewer transformer block reduces compute and parameters without changing batching or optimizer settings.

EVIDENCE: The best 8-layer design processed 486.2M tokens at 39.0% MFU and 50.3M parameters; after batch-size interpolation found 96 sequences best, removing one block is a controlled test of whether more training tokens outweigh modestly reduced depth.

<<<<<<< SEARCH
DEPTH = 8               # number of transformer layers
=======
DEPTH = 7               # preserve 512-wide model while removing one compute-heavy block
>>>>>>> REPLACE
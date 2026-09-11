MECHANISM: Lower-side fine-grained early-context interpolation

HYPOTHESIS: Using 452-token pre-global windows with 256-token post-global windows will reduce `val_bpb` below 0.983184 by trading a small amount of early context for greater token throughput.

INTENDED_EDIT: Narrow the first three local-attention windows from 456 to 452 tokens while retaining 256-token late windows, full attention at layers 4 and 8, and all optimizer settings.

EVIDENCE: The 456/256 design achieved 0.983184, outperforming 448/256 at 0.983556 and 464/256 at 0.983549; testing their lower-side midpoint refines the newly identified optimum while favoring throughput.

<<<<<<< SEARCH
        early_short_window = 57 * long_window // 256
=======
        early_short_window = 113 * long_window // 512
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 456 to 256
=======
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 452 to 256
>>>>>>> REPLACE
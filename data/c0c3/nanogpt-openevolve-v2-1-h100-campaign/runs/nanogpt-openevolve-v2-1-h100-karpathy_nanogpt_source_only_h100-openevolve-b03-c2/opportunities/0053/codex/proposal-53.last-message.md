MECHANISM: Fine-grained upper-side early-context interpolation

HYPOTHESIS: Increasing the three pre-global local windows from 464 to 480 tokens will reduce `val_bpb` below 0.983549 by moving slightly toward the broader-context regime without reaching the regressing 512-token endpoint.

INTENDED_EDIT: Set early local-attention windows to 480 tokens while retaining 256-token late windows, full attention at layers 4 and 8, and all optimizer settings.

EVIDENCE: The 464/256 design achieved 0.983549, narrowly outperforming 448/256 at 0.983556 and 512/256 at 0.983620; testing 480/256 probes the unexplored upper half of this bracket.

<<<<<<< SEARCH
        early_short_window = 29 * long_window // 128
=======
        early_short_window = 15 * long_window // 64
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 464 to 256
=======
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 480 to 256
>>>>>>> REPLACE
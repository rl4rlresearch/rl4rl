MECHANISM: Upper-side fine-grained early-context interpolation

HYPOTHESIS: Using 460-token pre-global windows with 256-token post-global windows will reduce `val_bpb` below 0.983184.

INTENDED_EDIT: Set the first three local-attention windows to 460 tokens while retaining 256-token late windows and full attention at layers 4 and 8.

EVIDENCE: The 456/256 design achieved the best `val_bpb` of 0.983184, while 464/256 reached 0.983549; their untested midpoint isolates the upper side of the current optimum.

<<<<<<< SEARCH
        early_short_window = 29 * long_window // 128
=======
        early_short_window = 115 * long_window // 512
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 464 to 256
=======
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 460 to 256
>>>>>>> REPLACE
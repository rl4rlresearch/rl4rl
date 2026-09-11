MECHANISM: Lower-side early-context interpolation

HYPOTHESIS: Using 456-token pre-global windows with 256-token post-global windows will reduce `val_bpb` below 0.983549.

INTENDED_EDIT: Widen the first three local-attention layers from 384 to 456 tokens while retaining 256-token late windows, full attention at layers 4 and 8, and all optimizer settings.

EVIDENCE: The 464/256 design achieved 0.983549 and 448/256 nearly matched it at 0.983556, while 480/256 regressed to 0.983946; testing their 456-token midpoint targets the lower side of the empirically favored bracket.

<<<<<<< SEARCH
        early_short_window = 3 * long_window // 16
        late_short_window = long_window // 8
=======
        early_short_window = 57 * long_window // 256
        late_short_window = long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 384 to 256
=======
WINDOW_PATTERN = "SSSL" # full context at layers 4/8; local windows narrow from 456 to 256
>>>>>>> REPLACE
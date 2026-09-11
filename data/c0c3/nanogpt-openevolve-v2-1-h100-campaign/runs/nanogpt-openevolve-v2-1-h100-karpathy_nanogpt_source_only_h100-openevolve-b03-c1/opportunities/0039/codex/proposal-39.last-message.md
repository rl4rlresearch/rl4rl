MECHANISM: Late-context window response-surface refinement

HYPOTHESIS: Increasing layer 6’s attention window from 1024 to 1280 tokens will reduce val_bpb below 0.984083 by moving toward the approximately 1200-token optimum interpolated from the measured 512-, 1024-, and 2048-token results.

INTENDED_EDIT: Change the single medium-context layer’s window from one-half to five-eighths of the 2048-token sequence while preserving all other settings.

EVIDENCE: Layer 6 windows of 512, 1024, and 2048 tokens produced val_bpb values of 0.984227, 0.984083, and 0.984308; quadratic interpolation places the local minimum near 1200 tokens.

<<<<<<< SEARCH
        medium_window = long_window // 2
=======
        medium_window = 5 * long_window // 8
>>>>>>> REPLACE
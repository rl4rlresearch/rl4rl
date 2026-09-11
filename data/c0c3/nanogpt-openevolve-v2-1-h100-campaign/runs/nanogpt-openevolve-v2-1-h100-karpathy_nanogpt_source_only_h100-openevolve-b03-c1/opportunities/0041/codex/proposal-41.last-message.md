MECHANISM: Asymmetric late-context window refinement

HYPOTHESIS: Reducing layer 6’s attention window from 1024 to 768 tokens will lower val_bpb below 0.984083 by moving toward the sub-1024 optimum implied by the measured response.

INTENDED_EDIT: Change only the single medium-context layer’s window from one-half to three-eighths of the 2048-token sequence.

EVIDENCE: Layer 6 at 1024 tokens achieved 0.984083, while increasing it to 1280 regressed sharply to 0.984367 and reducing it to 512 caused only a smaller regression to 0.984227; this asymmetric response motivates testing between 512 and 1024.

<<<<<<< SEARCH
        medium_window = long_window // 2
=======
        medium_window = 3 * long_window // 8
>>>>>>> REPLACE
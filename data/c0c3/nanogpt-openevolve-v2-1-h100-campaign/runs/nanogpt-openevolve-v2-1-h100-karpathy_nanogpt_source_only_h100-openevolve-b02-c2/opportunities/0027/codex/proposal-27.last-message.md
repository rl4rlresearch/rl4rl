MECHANISM: Bracketed local-attention window refinement

HYPOTHESIS: Using 192-token local windows will lower val_bpb below 0.983766 by retaining more useful local context than the unsuccessful 64-token design while remaining faster than the 256-token design.

INTENDED_EDIT: Set the six short-attention layers to 192 tokens while preserving full-context layers 4 and 8 and all optimizer settings.

EVIDENCE: The best result used 128-token windows at 0.983766; 64 tokens regressed sharply to 0.984784, while 256 tokens was only slightly worse at 0.983993, suggesting the optimum lies above 128 and motivating a hardware-aligned 192-token test.

<<<<<<< SEARCH
        short_window = 3 * long_window // 8
=======
        short_window = 3 * long_window // 32
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSL" # 192-token local layers with full context at layers 4 and 8
>>>>>>> REPLACE
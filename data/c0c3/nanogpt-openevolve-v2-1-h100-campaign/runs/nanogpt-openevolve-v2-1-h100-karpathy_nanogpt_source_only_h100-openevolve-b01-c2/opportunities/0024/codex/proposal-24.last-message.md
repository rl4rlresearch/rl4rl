MECHANISM: Bracketed 240-token local-attention refinement

HYPOTHESIS: Seven 240-token local layers followed by one full-context anchor will retain at least 523.8M training tokens and lower val_bpb below 0.992854.

INTENDED_EDIT: Use a single final global anchor and set the preceding seven local-attention windows to 240 tokens.

EVIDENCE: The 256-token design achieved the best val_bpb of 0.992854, while 224 tokens reached 0.992930; testing their midpoint is the most direct refinement of the bracketed optimum.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = 15 * long_window // 128
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # proven layout: two full-context anchor layers
=======
WINDOW_PATTERN = "SSSS" # seven 240-token local layers, then one forced full-context anchor
>>>>>>> REPLACE
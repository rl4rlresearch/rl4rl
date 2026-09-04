MECHANISM: Additional mid-depth global-attention integration

HYPOTHESIS: Adding a third full-context layer at layer 6 will lower val_bpb below 0.985746 because removing one of the baseline’s two global layers worsened validation to 0.986118 despite slightly higher throughput.

INTENDED_EDIT: Preserve the best optimizer and schedule while changing the eight-layer attention pattern from two to three full-context layers, located at layers 4, 6, and 8.

EVIDENCE: The two-global-layer baseline achieved 0.985746, while the otherwise matched single-global-layer design achieved 0.986118; this indicates global-context capacity was more valuable than its small throughput cost and motivates testing one additional global layer.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
=======
WINDOW_PATTERN = "SSSLSLSL" # full context at layers 4, 6, and 8
>>>>>>> REPLACE
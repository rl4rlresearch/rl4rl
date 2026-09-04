MECHANISM: Delayed intermediate global-context integration

HYPOTHESIS: Moving the intermediate full-context layer from layer 4 to layer 5 will reduce `val_bpb` below 0.983618 by allowing one additional 384-token local layer to build features before global mixing while retaining two post-integration refinement layers.

INTENDED_EDIT: Change the attention pattern from full context at layers 4 and 8 to full context at layers 5 and 8, preserving all window sizes and optimizer settings.

EVIDENCE: Moving intermediate global attention earlier to layer 3 regressed from 0.983618 to 0.984630, while removing it regressed to 0.985116; this motivates testing the opposite one-layer displacement while retaining the demonstrably necessary intermediate global layer.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=3/16 context
=======
WINDOW_PATTERN = "SSSSLSSL" # 384-token local layers; full context at layers 5 and 8
>>>>>>> REPLACE
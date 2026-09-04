MECHANISM: Later first global-context injection

HYPOTHESIS: Full attention at layers 5, 6, and 8 will beat 0.984244 val_bpb by concentrating global-context processing later while preserving the proven layer-6 and final-layer attention at unchanged compute cost.

INTENDED_EDIT: Move the earliest full-attention layer from depth 4 to depth 5, leaving all other architecture, optimizer, batching, and schedule settings unchanged.

EVIDENCE: Moving the earliest full-attention layer from depth 4 to depth 3 regressed val_bpb to 0.984720; testing depth 5 completes the nearest-neighbor placement comparison around the best depth-4 configuration.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSLSL" # full attention at layers 4, 6, and 8
=======
WINDOW_PATTERN = "SSSSLLSL" # full attention at layers 5, 6, and 8
>>>>>>> REPLACE
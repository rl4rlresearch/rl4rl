MECHANISM: Incremental global-attention density

HYPOTHESIS: Adding a fourth evenly spaced full-attention layer will beat 0.984244 val_bpb because validation improved as full-attention depth increased from one to two to three layers, despite the three-layer variant processing fewer tokens.

INTENDED_EDIT: Change the attention pattern so layers 2, 4, 6, and 8 use full attention while preserving all other settings.

EVIDENCE: One full-attention layer scored 0.985611, two scored 0.984313, and three scored 0.984244; the three-layer improvement occurred despite throughput falling from 506.2M to 499.6M tokens, indicating additional global-context capacity can outweigh its compute cost.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSLSL" # full attention at layers 4, 6, and 8
=======
WINDOW_PATTERN = "SLSLSLSL" # full attention at layers 2, 4, 6, and 8
>>>>>>> REPLACE
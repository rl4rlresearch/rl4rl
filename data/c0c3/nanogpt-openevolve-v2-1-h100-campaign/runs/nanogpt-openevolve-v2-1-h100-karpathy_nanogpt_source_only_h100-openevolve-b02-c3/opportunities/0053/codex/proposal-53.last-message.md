MECHANISM: Denser global anchors with narrower local attention

HYPOTHESIS: Halving short-layer windows to 256 tokens while retaining full attention in layers 4 and 8 will process at least 510M tokens and reduce val_bpb below 0.983505.

INTENDED_EDIT: Change the six short-attention layers from quarter-context to eighth-context windows without altering architecture, batch size, learning rates, or the two full-context layers.

EVIDENCE: The 212,992-token, 23/25-rate baseline reached 0.983505 after 498.6M tokens, while added attention specialization via eight heads regressed to 0.989288; narrower local windows test attention efficiency while preserving periodic global mixing.

<<<<<<< SEARCH
        short_window = long_window // 4
=======
        short_window = long_window // 8
>>>>>>> REPLACE

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=quarter context
=======
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=eighth context
>>>>>>> REPLACE
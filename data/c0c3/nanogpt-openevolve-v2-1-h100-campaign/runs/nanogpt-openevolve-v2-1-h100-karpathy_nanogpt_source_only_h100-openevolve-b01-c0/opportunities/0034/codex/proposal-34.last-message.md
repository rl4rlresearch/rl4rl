MECHANISM: Adjacent-layer global-attention placement

HYPOTHESIS: Moving the third full-attention layer from depth 6 to depth 5 will beat 0.984244 val_bpb if the benefit peaks earlier than the tested depth-6 placement.

INTENDED_EDIT: Use full attention at layers 4, 5, and 8 while preserving compute scale, optimizer, batching, and schedule.

EVIDENCE: Full attention at layers 4, 6, and 8 achieved 0.984244, while moving the added layer to depth 7 regressed to 0.985327; testing depth 5 completes the nearest-neighbor placement comparison around the successful depth-6 result.

<<<<<<< SEARCH
WINDOW_PATTERN = "SSSLSLSL" # full attention at layers 4, 6, and 8
=======
WINDOW_PATTERN = "SSSLLSSL" # full attention at layers 4, 5, and 8
>>>>>>> REPLACE
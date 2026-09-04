MECHANISM: Revised quadratic-interpolated linear warmdown

HYPOTHESIS: A 77.4% linear warmdown will beat 0.984455 val_bpb by placing cooldown near the local minimum fitted from the closest 75.6%, 78.5%, and 80% results.

INTENDED_EDIT: Reduce `WARMDOWN_RATIO` from 0.785 to 0.774 while preserving all other settings.

EVIDENCE: The 75.6%, 78.5%, and 80% runs achieved 0.984571, 0.984455, and 0.984745 val_bpb respectively; quadratic interpolation estimates their local minimum near 77.4%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.785  # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.774  # fraction of time budget for LR warmdown
>>>>>>> REPLACE
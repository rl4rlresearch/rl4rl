MECHANISM: Nearest-neighbor quadratic warmdown refinement

HYPOTHESIS: A 78.1% linear warmdown will beat 0.984313 val_bpb by placing cooldown near the local minimum fitted from the closest 77.4%, 78.0%, and 78.5% results.

INTENDED_EDIT: Increase `WARMDOWN_RATIO` from 0.780 to 0.781 while preserving all other architecture, optimizer, batching, and schedule settings.

EVIDENCE: The 77.4%, 78.0%, and 78.5% runs achieved 0.984918, 0.984313, and 0.984455 val_bpb respectively; quadratic interpolation of these nearest observations estimates the minimum near 78.1%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.780  # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.781  # fraction of time budget for LR warmdown
>>>>>>> REPLACE
MECHANISM: Quadratic-interpolated linear warmdown

HYPOTHESIS: A 78.5% linear warmdown will beat 0.984745 val_bpb by placing the cooldown near the local minimum implied by the 70%, 80%, and 90% results.

INTENDED_EDIT: Reduce `WARMDOWN_RATIO` from 0.8 to 0.785, delaying linear decay from 20% to 21.5% of the training window.

EVIDENCE: Val_bpb improved from 0.985071 at 70% warmdown to 0.984745 at 80%, then regressed to 0.985340 at 90%; quadratic interpolation of these three observations places the estimated minimum near 78.5%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.8    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.785  # fraction of time budget for LR warmdown
>>>>>>> REPLACE
MECHANISM: Fine-grained linear warmdown interpolation

HYPOTHESIS: A 78.0% linear warmdown will beat 0.984455 val_bpb by moving modestly from the empirical best toward the 77.4% local minimum estimated from nearby schedule results, without repeating the full 1.1-point shift that underperformed.

INTENDED_EDIT: Reduce `WARMDOWN_RATIO` from 0.785 to 0.780 while preserving the architecture, optimizer, batch size, and linear cooldown shape.

EVIDENCE: The best result used 78.5% warmdown (0.984455 val_bpb), while 75.6% remained close at 0.984571 and the fitted nearby minimum was 77.4%; testing 78.0% provides a tighter interpolation around the strongest observed region.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.785  # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.780  # fraction of time budget for LR warmdown
>>>>>>> REPLACE
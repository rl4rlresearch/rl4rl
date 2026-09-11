MECHANISM: Symmetric fine-grained warmdown refinement

HYPOTHESIS: A 77.9% linear warmdown will beat 0.984313 val_bpb by testing the lower-side neighbor of the best 78.0% setting after 78.1% regressed.

INTENDED_EDIT: Reduce `WARMDOWN_RATIO` from 0.780 to 0.779 while preserving all other settings.

EVIDENCE: The best result used 78.0% warmdown (0.984313), while the adjacent 78.1% test regressed to 0.985030; 77.9% is the most direct symmetric test of whether the local optimum lies just below 78.0%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.780  # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.779  # fraction of time budget for LR warmdown
>>>>>>> REPLACE
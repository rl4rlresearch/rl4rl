MECHANISM: Later-start zero-ending linear warmdown continuation

HYPOTHESIS: A 57% zero-ending linear warmdown with the proven 258,048-token batch will retain roughly 520M training tokens and lower `val_bpb` below 0.985148.

INTENDED_EDIT: Move linear-decay onset 1.5 percentage points later than the best verified 58.5% schedule while preserving all other hyperparameters.

EVIDENCE: Shortening warmdown from 63% through 61.5%, 60%, and 58.5% progressively improved `val_bpb` from 0.985414 to 0.985321, 0.985181, and 0.985148; the prior 57% attempt was not verified, so this continuation remains unresolved.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.6    # begin linear LR warmdown after 40% of the time budget
=======
WARMDOWN_RATIO = 0.57   # begin linear LR warmdown after 43% of the time budget
>>>>>>> REPLACE
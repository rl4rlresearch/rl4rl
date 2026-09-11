MECHANISM: Earlier linear learning-rate annealing

HYPOTHESIS: Extending linear warmdown from 60% to 70% of training will reduce val_bpb below 0.985492 by further limiting high-learning-rate updates during the middle of the fixed training window.

INTENDED_EDIT: Preserve the winning 112-sequence batch and all optimizer settings, but begin linear learning-rate decay at 30% rather than 40% of the training window.

EVIDENCE: Extending linear warmdown from 50% to 60% improved val_bpb from 0.985757 to 0.985492, while cosine annealing that retained larger early-decay learning rates regressed to 0.988710; testing 70% directly probes whether the measured benefit continues.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.6    # begin linear LR decay after 40% of the time budget
=======
WARMDOWN_RATIO = 0.7    # begin linear LR decay after 30% of the time budget
>>>>>>> REPLACE
MECHANISM: Earlier linear learning-rate annealing

HYPOTHESIS: Extending linear warmdown from 70% to 80% of training will reduce val_bpb below 0.984948 by further reducing high-learning-rate updates during the fixed training window.

INTENDED_EDIT: Preserve the winning 112-sequence batch and all optimizer settings, but begin linear learning-rate decay at 20% rather than 30% of the training window.

EVIDENCE: Extending linear warmdown from 50% to 60% improved val_bpb from 0.985757 to 0.985492, and extending it to 70% improved further to 0.984948, providing direct monotonic evidence for testing 80%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.7    # begin linear LR decay after 30% of the time budget
=======
WARMDOWN_RATIO = 0.8    # begin linear LR decay after 20% of the time budget
>>>>>>> REPLACE
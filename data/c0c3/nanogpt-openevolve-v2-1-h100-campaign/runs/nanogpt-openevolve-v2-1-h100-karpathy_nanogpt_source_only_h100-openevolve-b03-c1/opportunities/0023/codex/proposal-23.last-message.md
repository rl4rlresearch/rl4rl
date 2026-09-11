MECHANISM: Earlier linear learning-rate annealing

HYPOTHESIS: Extending linear warmdown from 80% to 90% of training will reduce val_bpb below 0.984506 by further limiting high-learning-rate updates during the fixed training window.

INTENDED_EDIT: Preserve the winning 112-sequence batch and all optimizer settings, but begin linear learning-rate decay at 10% rather than 20% of the training window.

EVIDENCE: Extending linear warmdown from 60% to 70% improved val_bpb from 0.985492 to 0.984948, and extending it to 80% improved further to 0.984506, providing direct monotonic evidence for testing 90%.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.8    # begin linear LR decay after 20% of the time budget
=======
WARMDOWN_RATIO = 0.9    # begin linear LR decay after 10% of the time budget
>>>>>>> REPLACE
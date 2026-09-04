MECHANISM: Earlier linear learning-rate annealing

HYPOTHESIS: Extending linear warmdown from 50% to 60% on the best 2.25×/2.75×/5.5× architecture will preserve roughly 500M-token throughput and reduce val_bpb below 0.991682.

INTENDED_EDIT: Start the proven linear decay at 40% rather than 50% of the training window while retaining the same peak and zero final learning rates.

EVIDENCE: Equal-duration cosine decay regressed sharply to 0.995010; because cosine maintains a higher learning rate through early warmdown than linear decay, this motivates testing earlier linear annealing without changing the best architecture.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.6    # fraction of time budget for LR warmdown
>>>>>>> REPLACE
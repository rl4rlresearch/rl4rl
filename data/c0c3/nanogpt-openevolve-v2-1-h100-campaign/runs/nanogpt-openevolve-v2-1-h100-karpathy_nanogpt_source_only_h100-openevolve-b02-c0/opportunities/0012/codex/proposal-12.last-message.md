MECHANISM: Earlier linear learning-rate annealing with preserved peak rates

HYPOTHESIS: Extending warmdown from 50% to 75% will reduce val_bpb below 0.987174 by lowering mid-to-late optimization noise while retaining the winning peak learning rates during the first quarter of training.

INTENDED_EDIT: Begin linear learning-rate decay after 25% of the training window instead of 50%, leaving architecture, batch size, and peak rates unchanged.

EVIDENCE: Delaying warmdown to the final 25% substantially worsened val_bpb to 0.991963, while uniformly reducing learning rates by 29% produced a near-tie at 0.987410; this motivates a moderate reduction in cumulative learning rate that preserves full-rate early training.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.75   # fraction of time budget for LR warmdown
>>>>>>> REPLACE
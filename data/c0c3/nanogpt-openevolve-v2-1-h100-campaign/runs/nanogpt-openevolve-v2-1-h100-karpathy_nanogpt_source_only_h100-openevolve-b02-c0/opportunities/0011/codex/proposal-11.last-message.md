MECHANISM: Delayed linear learning-rate warmdown

HYPOTHESIS: Shortening warmdown from 50% to 25% will beat `val_bpb` 0.987174 by preserving higher learning rates through mid-training while still annealing to zero before validation.

INTENDED_EDIT: Begin the existing linear learning-rate decay after 75% of the training window instead of after 50%, without changing architecture, batch size, peak learning rates, or throughput.

EVIDENCE: Uniformly scaling learning rates down by 29% preserved throughput but worsened `val_bpb` from 0.987174 to 0.987410, indicating that the winning 262K-token configuration is more likely under-updated than overstepping.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.25   # fraction of time budget for LR warmdown
>>>>>>> REPLACE
MECHANISM: Delayed learning-rate annealing

HYPOTHESIS: Starting warmdown at 60% rather than 50% of training will increase useful high-rate optimization while retaining a two-minute decay, lowering val_bpb below 0.995558 without reducing throughput.

INTENDED_EDIT: Reduce WARMDOWN_RATIO from 0.5 to 0.4 while preserving the baseline architecture, batching, and learning rates.

EVIDENCE: Every attention modification reduced training tokens or failed to improve validation; the closest native-dispatch result reached 0.995868 with similar throughput, motivating an isolated optimization-schedule change.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.4    # fraction of time budget for LR warmdown
>>>>>>> REPLACE
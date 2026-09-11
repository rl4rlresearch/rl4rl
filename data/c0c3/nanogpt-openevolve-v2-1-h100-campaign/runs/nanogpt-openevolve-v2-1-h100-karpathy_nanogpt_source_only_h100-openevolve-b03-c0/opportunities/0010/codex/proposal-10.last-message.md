MECHANISM: Shorter terminal learning-rate decay

HYPOTHESIS: Extending the full-learning-rate phase from 50% to 60% of training will achieve `val_bpb < 0.986636` without reducing the best design’s token throughput or update count.

INTENDED_EDIT: Reduce the linear warmdown duration from 50% to 40% of the fixed training window.

EVIDENCE: The 96-sequence design is the strongest tested configuration, while every tested depth or MLP-capacity change regressed; its improvement from denser updates indicates optimization remains consequential, motivating an isolated schedule change that preserves its architecture and efficient batch shape.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.4    # fraction of time budget for LR warmdown
>>>>>>> REPLACE
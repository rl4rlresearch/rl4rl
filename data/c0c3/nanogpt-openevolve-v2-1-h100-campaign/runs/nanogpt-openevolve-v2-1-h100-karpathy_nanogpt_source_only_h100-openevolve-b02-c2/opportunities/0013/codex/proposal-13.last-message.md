MECHANISM: Shortened learning-rate warmdown

HYPOTHESIS: Delaying LR decay from 50% to 60% of the training window will lower val_bpb below 0.985746 by adding roughly 187 full-rate updates while retaining a two-minute convergence phase.

INTENDED_EDIT: Shorten the time-based learning-rate warmdown from 50% to 40%, leaving the best 262K-token batch and Muon weight decay unchanged.

EVIDENCE: The best design reaches 0.985746 in 1,874 steps, while decay refinements on both sides of weight decay 0.1 were worse; this motivates holding the bracketed decay setting fixed and testing whether the short run benefits from more full-rate optimization.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.4    # preserve full LR longer while retaining a two-minute warmdown
>>>>>>> REPLACE
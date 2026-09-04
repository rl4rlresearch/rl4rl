MECHANISM: Earlier linear learning-rate annealing

HYPOTHESIS: Extending warmdown from 50% to 60% of training will achieve `val_bpb < 0.986636` by improving late-stage convergence without changing throughput or model capacity.

INTENDED_EDIT: Increase `WARMDOWN_RATIO` from `0.5` to `0.6`, starting linear decay at 40% of the fixed training window.

EVIDENCE: Shortening warmdown to 40% regressed `val_bpb` from `0.986636` to `0.988601` with essentially unchanged throughput, providing directional evidence that the best 96-sequence design benefits from more—not less—annealing.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WARMDOWN_RATIO = 0.6    # fraction of time budget for LR warmdown
>>>>>>> REPLACE
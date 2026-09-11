MECHANISM: Earlier linear learning-rate warmdown

HYPOTHESIS: Starting warmdown at 40% rather than 50% of the training window will reduce late-stage optimization noise at the successful 262K-token batch size and lower val_bpb below 0.986167.

INTENDED_EDIT: Increase WARMDOWN_RATIO from 0.5 to 0.6 while preserving the model, batch size, peak learning rates, and zero final learning rate.

EVIDENCE: Halving the batch from 524K to 262K improved val_bpb from 0.992286 to 0.986167, but halving it again to 131K regressed to 0.991527, indicating excessive high-frequency optimization is harmful; a slightly earlier warmdown tests reduced late update magnitude without sacrificing throughput or update cadence.

<<<<<<< SEARCH
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
=======
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.6    # fraction of time budget for LR warmdown
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
>>>>>>> REPLACE
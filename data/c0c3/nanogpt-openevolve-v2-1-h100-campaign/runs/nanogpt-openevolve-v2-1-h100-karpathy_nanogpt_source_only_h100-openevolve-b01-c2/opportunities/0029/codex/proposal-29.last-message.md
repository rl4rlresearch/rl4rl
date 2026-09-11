MECHANISM: Earlier linear warmdown on the proven 256-token attention baseline

HYPOTHESIS: Extending linear warmdown from 60% to 70% of training will retain roughly 528M tokens while lowering val_bpb below 0.992358.

INTENDED_EDIT: Preserve the current architecture, optimizer, and batching while starting linear LR decay at 30% instead of 40% of the training window.

EVIDENCE: Extending linear warmdown from 50% to 60% improved val_bpb from 0.992854 to 0.992358 without reducing throughput, whereas cosine decay regressed to 0.995509; this directly tests whether the beneficial earlier linear decay trend continues.

<<<<<<< SEARCH
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.6    # begin linear LR warmdown after 40% of the time budget
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
=======
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.7    # begin linear LR warmdown after 30% of the time budget
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
>>>>>>> REPLACE
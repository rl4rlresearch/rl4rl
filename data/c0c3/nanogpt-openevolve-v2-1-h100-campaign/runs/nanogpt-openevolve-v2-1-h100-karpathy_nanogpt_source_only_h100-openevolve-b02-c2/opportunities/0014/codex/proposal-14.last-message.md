MECHANISM: Extended learning-rate convergence warmdown

HYPOTHESIS: Restoring Muon weight decay to the best verified 0.1 and extending LR warmdown from 50% to 60% will lower val_bpb below 0.985746 by beginning convergence earlier; shortening warmdown to 40% worsened val_bpb to 0.987004.

INTENDED_EDIT: Restore the best cautious-decay setting and increase WARMDOWN_RATIO to 0.6 while preserving the proven 262K-token configuration.

EVIDENCE: Weight decay 0.1 achieved the best val_bpb of 0.985746, while nearby 0.095 and 0.125 produced 0.986110 and 0.986461. At weight decay 0.1, shortening warmdown to 40% worsened validation, motivating a symmetric test of a longer warmdown.

<<<<<<< SEARCH
WEIGHT_DECAY = 0.095    # refine the observed cautious-decay optimum near 0.1
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.5    # fraction of time budget for LR warmdown
=======
WEIGHT_DECAY = 0.1      # best verified cautious-decay setting
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.6    # begin LR convergence earlier after shorter warmdown regressed
>>>>>>> REPLACE
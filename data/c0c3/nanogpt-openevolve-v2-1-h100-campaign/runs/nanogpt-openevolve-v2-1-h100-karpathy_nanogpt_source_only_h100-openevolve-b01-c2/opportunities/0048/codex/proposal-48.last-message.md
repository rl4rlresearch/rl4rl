MECHANISM: Nonzero linear-decay learning-rate tail

HYPOTHESIS: Restoring `MATRIX_LR=0.040` and ending the proven linear warmdown at 5% of peak LR will retain roughly 520M tokens while lowering `val_bpb` below 0.985181.

INTENDED_EDIT: Restore the best Muon learning rate and add a modest 5% final LR floor, preserving the architecture, 258,048-token batch, and 60% linear warmdown.

EVIDENCE: The `MATRIX_LR=0.040` design achieved the best `val_bpb` of 0.985181. Equal-area cosine warmdown regressed to 0.988030 while shifting LR exposure from the latter half of decay toward the earlier half, motivating a direct test of preserving more late-stage learning rate with the successful linear shape.

<<<<<<< SEARCH
MATRIX_LR = 0.038       # 5% lower Muon learning rate at the best batch size
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.6    # begin linear LR warmdown after 40% of the time budget
FINAL_LR_FRAC = 0.0     # final LR as fraction of initial
=======
MATRIX_LR = 0.04        # best verified learning rate for matrix parameters (Muon)
SCALAR_LR = 0.5         # learning rate for per-layer scalars (Adam)
WEIGHT_DECAY = 0.2      # cautious weight decay for Muon
ADAM_BETAS = (0.8, 0.95) # Adam beta1, beta2
WARMUP_RATIO = 0.0      # fraction of time budget for LR warmup
WARMDOWN_RATIO = 0.6    # begin linear LR warmdown after 40% of the time budget
FINAL_LR_FRAC = 0.05    # preserve a small late-stage optimization tail
>>>>>>> REPLACE
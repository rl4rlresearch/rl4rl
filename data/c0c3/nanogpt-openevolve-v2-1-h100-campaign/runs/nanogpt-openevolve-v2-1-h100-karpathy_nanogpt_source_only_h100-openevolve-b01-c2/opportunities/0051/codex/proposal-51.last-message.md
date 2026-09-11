MECHANISM: Symmetric later-start linear warmdown refinement

HYPOTHESIS: A 58.5% zero-ending linear warmdown with the proven 258,048-token batch will retain roughly 520M training tokens and lower `val_bpb` below 0.985181.

INTENDED_EDIT: Restore a zero final learning rate and move the linear-decay start 1.5 percentage points later, preserving all peak learning rates, optimizer settings, and architecture.

EVIDENCE: The 60% zero-ending warmdown achieved the best `val_bpb` of 0.985181, while moving its start 1.5 points earlier regressed modestly to 0.985321; testing the symmetric later-start schedule directly resolves the untested higher-exposure side without the endpoint change that confounded the 5% LR-floor result.

<<<<<<< SEARCH
WARMDOWN_RATIO = 0.6    # begin linear LR warmdown after 40% of the time budget
FINAL_LR_FRAC = 0.05    # preserve a small late-stage optimization tail
=======
WARMDOWN_RATIO = 0.585  # begin linear LR warmdown after 41.5% of the time budget
FINAL_LR_FRAC = 0.0     # decay to zero at the end of training
>>>>>>> REPLACE
MECHANISM: Tighter terminal-checkpoint weight averaging

HYPOTHESIS: Averaging snapshots from only the final 2.5% of training will exceed 9,252 correct predictions, or tie while lowering cross-entropy, by further reducing mismatch between averaged parameters and final BatchNorm statistics.

INTENDED_EDIT: Move the start of sparse parameter averaging from 95% to 97.5% of the trajectory while preserving its cadence and all other behavior.

EVIDENCE: Narrowing the averaging window from the final 10% to 5% retained 9,252 correct predictions and lowered cross-entropy from 0.2360583 to 0.2356299, while averaging BatchNorm statistics hurt accuracy; this motivates another controlled narrowing of parameter averaging alone.

<<<<<<< SEARCH
    if progress >= 0.95 and (step % 4 == 0 or is_final_step):
=======
    if progress >= 0.975 and (step % 4 == 0 or is_final_step):
>>>>>>> REPLACE
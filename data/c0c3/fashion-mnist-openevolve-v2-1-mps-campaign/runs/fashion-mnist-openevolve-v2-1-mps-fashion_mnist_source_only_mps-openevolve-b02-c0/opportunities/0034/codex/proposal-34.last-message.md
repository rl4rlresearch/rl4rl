MECHANISM: Narrow late-checkpoint weight averaging

HYPOTHESIS: Averaging snapshots from only the final 5% of training will exceed 9,252 correct predictions by retaining the variance reduction of successful weight averaging while reducing bias from earlier parameters and mismatch with final BatchNorm statistics.

INTENDED_EDIT: Move the start of sparse parameter averaging from 90% to 95% of the trajectory, preserving its cadence and all other training and inference behavior.

EVIDENCE: Final-10% parameter averaging improved validation_correct from 9,249 to 9,252, whereas also averaging BatchNorm statistics reduced it to 9,248; narrowing the parameter window is a targeted way to align averaged weights more closely with the beneficial final BatchNorm state.

<<<<<<< SEARCH
    if progress >= 0.90 and (step % 4 == 0 or is_final_step):
=======
    if progress >= 0.95 and (step % 4 == 0 or is_final_step):
>>>>>>> REPLACE
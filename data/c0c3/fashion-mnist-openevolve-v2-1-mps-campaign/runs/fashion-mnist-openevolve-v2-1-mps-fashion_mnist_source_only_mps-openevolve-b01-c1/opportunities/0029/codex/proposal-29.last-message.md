MECHANISM: Bracketed ensemble-loss weight refinement

HYPOTHESIS: An ensemble-loss weight of 0.70 will exceed 9,236 correct predictions by preserving stronger individual-view regularization than 0.75 while retaining most of the gain from ensemble-aligned supervision.

INTENDED_EDIT: Reweight the six-view objective from 25% individual and 75% ensemble loss to 30% individual and 70% ensemble loss.

EVIDENCE: Increasing the ensemble weight from 0.50 to 0.75 raised correctness from 9,218 to 9,236, but further increases to 0.80 and 0.875 reduced it to 9,230 and 9,232; testing 0.70 probes the untested side of the apparent optimum.

<<<<<<< SEARCH
    return 0.25 * individual_loss + 0.75 * ensemble_loss
=======
    return 0.30 * individual_loss + 0.70 * ensemble_loss
>>>>>>> REPLACE
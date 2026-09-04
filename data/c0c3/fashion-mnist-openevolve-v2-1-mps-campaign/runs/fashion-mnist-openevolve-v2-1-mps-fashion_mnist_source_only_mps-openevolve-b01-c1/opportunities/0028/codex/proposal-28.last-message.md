MECHANISM: Intermediate deployed-ensemble loss weighting

HYPOTHESIS: An ensemble-loss weight of 0.80 will exceed 9,236 correct predictions by increasing validation-aligned supervision modestly without the accuracy loss observed at 0.875.

INTENDED_EDIT: Reweight the six-view objective from 25% individual and 75% ensemble loss to 20% individual and 80% ensemble loss.

EVIDENCE: Raising the ensemble weight from 0.50 to 0.75 improved correctness from 9,218 to 9,236, while raising it further to 0.875 reduced correctness to 9,232; testing 0.80 targets the supported interval near the observed optimum.

<<<<<<< SEARCH
    return 0.25 * individual_loss + 0.75 * ensemble_loss
=======
    return 0.20 * individual_loss + 0.80 * ensemble_loss
>>>>>>> REPLACE
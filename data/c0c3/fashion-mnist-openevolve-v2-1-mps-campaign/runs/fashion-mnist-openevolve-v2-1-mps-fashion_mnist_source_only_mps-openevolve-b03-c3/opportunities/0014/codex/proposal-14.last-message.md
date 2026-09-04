MECHANISM: Flip-ensemble-dominant supervision

HYPOTHESIS: Increasing the flip-averaged loss weight from 0.5 to 0.75 will exceed 9,229 correct predictions by emphasizing the decision rule used at validation while retaining enough individual-view supervision for robustness.

INTENDED_EDIT: Reweight the existing loss mixture to 25% individual-view cross-entropy and 75% flip-averaged cross-entropy.

EVIDENCE: Adding equal-weight flip-ensemble supervision improved the differential-basis model from 9,224 to 9,229 correct; increasing its weight is the most direct isolated test of whether stronger train/evaluation alignment yields further gains.

<<<<<<< SEARCH
    return 0.5 * individual_loss + 0.5 * ensemble_loss
=======
    return 0.25 * individual_loss + 0.75 * ensemble_loss
>>>>>>> REPLACE
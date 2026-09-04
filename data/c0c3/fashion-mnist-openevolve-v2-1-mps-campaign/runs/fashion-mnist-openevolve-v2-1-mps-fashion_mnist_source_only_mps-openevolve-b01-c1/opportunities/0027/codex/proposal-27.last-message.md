MECHANISM: Higher-weight deployed-ensemble supervision

HYPOTHESIS: Increasing the ensemble-loss weight to 0.875 while retaining 0.125 individual-view supervision will exceed 9,236 correct predictions by further aligning gradients with validation-time averaging without discarding per-view regularization.

INTENDED_EDIT: Reweight the existing six-view objective from 25% individual and 75% ensemble loss to 12.5% individual and 87.5% ensemble loss.

EVIDENCE: Raising the ensemble weight from 0.5 to 0.75 improved validation correctness from 9,218 to 9,236, the strongest current result, indicating that further emphasis on the deployed ensemble is the most promising next test.

<<<<<<< SEARCH
    return 0.25 * individual_loss + 0.75 * ensemble_loss
=======
    return 0.125 * individual_loss + 0.875 * ensemble_loss
>>>>>>> REPLACE
MECHANISM: Stronger deployed-ensemble supervision

HYPOTHESIS: Increasing the ensemble-loss weight from 0.5 to 0.75 will exceed 9,218 correct predictions by concentrating optimization on the averaged logits used at validation while retaining enough per-view supervision for stable features.

INTENDED_EDIT: Reweight the existing six-view objective to 25% individual-view cross-entropy and 75% evaluation-aligned ensemble cross-entropy.

EVIDENCE: Ensemble-aligned supervision produced successive gains from 9,063 to 9,124, 9,209, and 9,218 correct; expanding central sampling did not improve further, motivating stronger use of the already successful ensemble objective instead of adding views.

<<<<<<< SEARCH
    return 0.5 * individual_loss + 0.5 * ensemble_loss
=======
    return 0.25 * individual_loss + 0.75 * ensemble_loss
>>>>>>> REPLACE
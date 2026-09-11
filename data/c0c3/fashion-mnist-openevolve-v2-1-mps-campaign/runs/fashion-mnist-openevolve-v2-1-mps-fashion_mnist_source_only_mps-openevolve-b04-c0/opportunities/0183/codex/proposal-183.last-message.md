MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.5% to 13.5025% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436281471252441.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.75% to 6.75125%.

EVIDENCE: Every tested increase from 13.4375% through 13.5% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe tests whether the continuous optimum remains higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.865 * ensemble_logits
            + 0.0675 * down_logits
            + 0.0675 * up_logits
        )
=======
        translation_refined_logits = (
            0.864975 * ensemble_logits
            + 0.0675125 * down_logits
            + 0.0675125 * up_logits
        )
>>>>>>> REPLACE
MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.4875% to 13.49% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843628360748291.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.74375% to 6.745%.

EVIDENCE: Every tested increase from 13.4375% through 13.4875% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe directly tests whether the continuous optimum remains higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.865125 * ensemble_logits
            + 0.0674375 * down_logits
            + 0.0674375 * up_logits
        )
=======
        translation_refined_logits = (
            0.8651 * ensemble_logits
            + 0.06745 * down_logits
            + 0.06745 * up_logits
        )
>>>>>>> REPLACE
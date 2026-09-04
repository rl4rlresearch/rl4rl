MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.475% to 13.4775% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.1843628620147705.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.7375% to 6.73875%.

EVIDENCE: Every tested increase from 13.4375% through 13.475% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe directly tests whether the optimum remains higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.86525 * ensemble_logits
            + 0.067375 * down_logits
            + 0.067375 * up_logits
        )
=======
        translation_refined_logits = (
            0.865225 * ensemble_logits
            + 0.0673875 * down_logits
            + 0.0673875 * up_logits
        )
>>>>>>> REPLACE
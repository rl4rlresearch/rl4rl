MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.4925% to 13.495% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436282691955566.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.74625% to 6.7475%.

EVIDENCE: Every tested increase from 13.4375% through 13.4925% preserved correctness and reduced cross-entropy, so another equal-sized one-sided probe directly tests whether the continuous optimum remains higher.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.865075 * ensemble_logits
            + 0.0674625 * down_logits
            + 0.0674625 * up_logits
        )
=======
        translation_refined_logits = (
            0.86505 * ensemble_logits
            + 0.067475 * down_logits
            + 0.067475 * up_logits
        )
>>>>>>> REPLACE
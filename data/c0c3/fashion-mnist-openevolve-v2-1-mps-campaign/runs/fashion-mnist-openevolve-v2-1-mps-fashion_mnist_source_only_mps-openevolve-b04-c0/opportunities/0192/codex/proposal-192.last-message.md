MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.5225% to 13.525% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436277389526368.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.76125% to 6.7625%.

EVIDENCE: Every tested increase from 13.4375% through 13.5225% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.864775 * ensemble_logits
            + 0.0676125 * down_logits
            + 0.0676125 * up_logits
        )
=======
        translation_refined_logits = (
            0.86475 * ensemble_logits
            + 0.067625 * down_logits
            + 0.067625 * up_logits
        )
>>>>>>> REPLACE
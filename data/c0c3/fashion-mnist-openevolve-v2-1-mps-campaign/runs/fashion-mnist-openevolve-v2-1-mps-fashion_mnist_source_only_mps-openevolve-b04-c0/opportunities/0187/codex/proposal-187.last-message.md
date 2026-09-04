MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.51% to 13.5125% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436280059814453.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.755% to 6.75625%.

EVIDENCE: Every tested increase from 13.4375% through 13.51% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.8649 * ensemble_logits
            + 0.06755 * down_logits
            + 0.06755 * up_logits
        )
=======
        translation_refined_logits = (
            0.864875 * ensemble_logits
            + 0.0675625 * down_logits
            + 0.0675625 * up_logits
        )
>>>>>>> REPLACE
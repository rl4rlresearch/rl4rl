MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.505% to 13.5075% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436280670166016.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.7525% to 6.75375%.

EVIDENCE: Every tested increase from 13.4375% through 13.505% preserved all 9,359 correct predictions and reduced cross-entropy, supporting another equal-sized one-sided probe.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.86495 * ensemble_logits
            + 0.067525 * down_logits
            + 0.067525 * up_logits
        )
=======
        translation_refined_logits = (
            0.864925 * ensemble_logits
            + 0.0675375 * down_logits
            + 0.0675375 * up_logits
        )
>>>>>>> REPLACE
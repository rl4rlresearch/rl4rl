MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.5025% to 13.505% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436280975341796.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.75125% to 6.7525%.

EVIDENCE: Every tested increase from 13.4375% through 13.5025% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.864975 * ensemble_logits
            + 0.0675125 * down_logits
            + 0.0675125 * up_logits
        )
=======
        translation_refined_logits = (
            0.86495 * ensemble_logits
            + 0.067525 * down_logits
            + 0.067525 * up_logits
        )
>>>>>>> REPLACE
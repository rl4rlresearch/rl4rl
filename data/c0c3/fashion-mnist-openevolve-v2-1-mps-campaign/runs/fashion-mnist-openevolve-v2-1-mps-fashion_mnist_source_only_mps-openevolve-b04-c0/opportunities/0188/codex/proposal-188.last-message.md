MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.5125% to 13.515% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436279525756835.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.75625% to 6.7575%.

EVIDENCE: Every tested increase from 13.4375% through 13.5125% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.864875 * ensemble_logits
            + 0.0675625 * down_logits
            + 0.0675625 * up_logits
        )
=======
        translation_refined_logits = (
            0.86485 * ensemble_logits
            + 0.067575 * down_logits
            + 0.067575 * up_logits
        )
>>>>>>> REPLACE
MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.515% to 13.5175% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436279106140135.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.7575% to 6.75875%.

EVIDENCE: Every tested increase from 13.4375% through 13.515% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.86485 * ensemble_logits
            + 0.067575 * down_logits
            + 0.067575 * up_logits
        )
=======
        translation_refined_logits = (
            0.864825 * ensemble_logits
            + 0.0675875 * down_logits
            + 0.0675875 * up_logits
        )
>>>>>>> REPLACE
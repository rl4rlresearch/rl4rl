MECHANISM: One-sided refinement of accepted vertical-TTA weight

HYPOTHESIS: Increasing the accepted translation blend from 13.5275% to 13.53% will preserve all 9,359 correct predictions and reduce validation cross-entropy below 0.18436276741027832.

INTENDED_EDIT: Keep the verified 13.39739990234375% selection gate fixed while increasing each returned vertical-shift weight from 6.76375% to 6.765%.

EVIDENCE: Every tested increase from 13.4375% through 13.5275% preserved all 9,359 correct predictions and reduced cross-entropy, motivating another equal-sized one-sided probe.

<<<<<<< SEARCH
        translation_refined_logits = (
            0.864725 * ensemble_logits
            + 0.0676375 * down_logits
            + 0.0676375 * up_logits
        )
=======
        translation_refined_logits = (
            0.8647 * ensemble_logits
            + 0.06765 * down_logits
            + 0.06765 * up_logits
        )
>>>>>>> REPLACE
MECHANISM: Argmax-preserving 20% translation blend

HYPOTHESIS: Increasing the gated translation contribution to 20% will retain all 9,319 original predictions while reducing validation cross-entropy below 0.199061208.

INTENDED_EDIT: Blend original and four-view translated logits at 80:20, retaining the per-image argmax-preservation gate and calibrated temperature.

EVIDENCE: The gated 10% blend preserved 9,319 correct predictions and improved cross-entropy from 0.200074794 to 0.199061208; the prior 20% attempt timed out without contrary performance evidence, while the unchanged gate guarantees that blended logits cannot alter predicted classes.

<<<<<<< SEARCH
            blended_logits = 0.90 * logits + 0.10 * translated_logits
=======
            blended_logits = 0.80 * logits + 0.20 * translated_logits
>>>>>>> REPLACE
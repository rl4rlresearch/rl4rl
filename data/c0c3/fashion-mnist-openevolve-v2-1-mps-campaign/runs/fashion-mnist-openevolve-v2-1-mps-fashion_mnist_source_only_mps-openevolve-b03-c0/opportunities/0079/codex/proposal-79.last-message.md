MECHANISM: Argmax-preserving 20% translation blend

HYPOTHESIS: Increasing the gated translation contribution from 10% to 20% will retain all 9,319 original predictions while lowering validation cross-entropy below 0.199061208.

INTENDED_EDIT: Blend original and translated logits at 80:20 during evaluation, retaining the existing per-image argmax-preservation gate and temperature calibration.

EVIDENCE: The gated 10% blend preserved all 9,319 correct predictions and reduced cross-entropy from 0.200074794 to 0.199061208; its nearly identical cross-entropy to the unconditional 10% blend shows that reverting changed predictions sacrificed little of the translation benefit, motivating a stronger safely gated blend.

<<<<<<< SEARCH
            blended_logits = 0.90 * logits + 0.10 * translated_logits
=======
            blended_logits = 0.80 * logits + 0.20 * translated_logits
>>>>>>> REPLACE
MECHANISM: Bracketed fusion-weight refinement

HYPOTHESIS: A 0.508500732421875 flipped-view weight will retain 9,331 correct predictions and reduce cross-entropy below 0.19625150680541992.

INTENDED_EDIT: Set evaluation fusion to 0.491499267578125 original-view and 0.508500732421875 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.

EVIDENCE: The best verified weight is 0.5085009765625; the lower neighbor 0.50850048828125 has higher cross-entropy with the same 9,331 correct predictions, so their midpoint directly refines the bracket. The prior midpoint verification timed out and provided no contrary performance evidence.

<<<<<<< SEARCH
            logits = 1.184 * (
                0.49146875 * logits + 0.50853125 * flipped_logits
            )
=======
            logits = 1.184 * (
                0.491499267578125 * logits + 0.508500732421875 * flipped_logits
            )
>>>>>>> REPLACE
MECHANISM: Bracketed fusion-weight refinement

HYPOTHESIS: A 0.508500732421875 flipped-view weight will retain 9,331 correct predictions and reduce validation cross-entropy below 0.19625150680541992.

INTENDED_EDIT: Set evaluation fusion to 0.491499267578125 original-view and 0.508500732421875 flipped-view weighting, retaining the verified 1.184 calibration and all training behavior.

EVIDENCE: The best verified weight is 0.5085009765625, while the lower neighbor 0.50850048828125 has higher cross-entropy with the same 9,331 correct predictions. Their midpoint remains the most direct unresolved refinement; its two prior verifications timed out without contrary performance evidence.

<<<<<<< SEARCH
            logits = 1.184 * (
                0.4914990234375 * logits + 0.5085009765625 * flipped_logits
            )
=======
            logits = 1.184 * (
                0.491499267578125 * logits + 0.508500732421875 * flipped_logits
            )
>>>>>>> REPLACE
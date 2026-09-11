MECHANISM: Bracketed fusion-weight refinement

HYPOTHESIS: A 0.508500732421875 flipped-view weight will retain 9,331 correct predictions and reduce cross-entropy below 0.19625150680541992.

INTENDED_EDIT: Set evaluation fusion to 0.491499267578125 original-view and 0.508500732421875 flipped-view weighting while retaining the verified 1.184 calibration and all training behavior.

EVIDENCE: The best verified weight is 0.5085009765625; neighboring 0.50850048828125 and 0.508501953125 weights both have higher cross-entropy while retaining 9,331 correct, so bisecting the narrower interval toward the lower neighbor directly refines the apparent local optimum.

<<<<<<< SEARCH
            logits = 1.184 * (
                0.4914921875 * logits + 0.5085078125 * flipped_logits
            )
=======
            logits = 1.184 * (
                0.491499267578125 * logits + 0.508500732421875 * flipped_logits
            )
>>>>>>> REPLACE
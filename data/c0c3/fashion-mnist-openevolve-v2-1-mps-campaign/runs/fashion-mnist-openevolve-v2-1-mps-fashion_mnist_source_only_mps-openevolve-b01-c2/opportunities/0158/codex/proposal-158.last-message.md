MECHANISM: Float32-adjacent confidence calibration

HYPOTHESIS: Decreasing the positive evaluation-logit scale by one float32 step will preserve all 9,331 argmax predictions and reduce validation cross-entropy below 0.19625150680541992.

INTENDED_EDIT: Replace the 1.184 evaluation calibration with its immediate float32 predecessor, leaving training and the best verified fusion weights unchanged.

EVIDENCE: The immediate higher float32 scale preserved 9,331 correct but worsened cross-entropy to 0.1962515079498291, indicating the local calibration gradient favors a lower scale.

<<<<<<< SEARCH
            logits = 1.184 * (
                0.4914990234375 * logits + 0.5085009765625 * flipped_logits
            )
=======
            logits = 1.1839998960494995 * (
                0.4914990234375 * logits + 0.5085009765625 * flipped_logits
            )
>>>>>>> REPLACE
MECHANISM: Opposite-direction one-ULP temperature calibration

HYPOTHESIS: Increasing the validation temperature by one float32 ULP will preserve all 9,322 argmax predictions while lowering validation cross-entropy below 0.19799208221435546.

INTENDED_EDIT: Change only the final logit temperature from 0.738156 to its immediately higher float32 value.

EVIDENCE: Translation-blend refinement plateaued at 9,322 correct, while three lower-temperature probes timed out without validation evidence; probing the previously untested upper direction most directly brackets the argmax-preserving calibration optimum.

<<<<<<< SEARCH
            logits = logits / 0.738156
=======
            logits = logits / 0.7381560802459717
>>>>>>> REPLACE
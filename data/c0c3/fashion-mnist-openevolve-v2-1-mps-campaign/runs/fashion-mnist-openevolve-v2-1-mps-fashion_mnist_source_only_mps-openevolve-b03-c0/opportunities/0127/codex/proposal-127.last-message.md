MECHANISM: Three-ULP temperature calibration

HYPOTHESIS: Decreasing the positive validation temperature by three float32 ULPs will preserve all 9,322 argmax predictions while lowering validation cross-entropy below 0.19799208221435546.

INTENDED_EDIT: Change only the final logit temperature from 0.738156 to its third-lower float32 value.

EVIDENCE: Translation-blend refinement plateaued at 9,322 correct, while the one- and two-ULP lower-temperature probes timed out without contrary validation evidence; a third-ULP decrement continues the argmax-preserving calibration search.

<<<<<<< SEARCH
            logits = logits / 0.738156
=======
            logits = logits / 0.7381558418273926
>>>>>>> REPLACE
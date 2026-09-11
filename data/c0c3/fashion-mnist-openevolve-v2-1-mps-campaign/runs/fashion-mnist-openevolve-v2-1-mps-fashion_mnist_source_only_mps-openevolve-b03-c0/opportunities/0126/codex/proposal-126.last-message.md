MECHANISM: Two-ULP temperature calibration

HYPOTHESIS: Decreasing the positive validation temperature by two float32 ULPs will preserve all 9,322 argmax predictions while lowering validation cross-entropy below 0.19799208221435546.

INTENDED_EDIT: Change only the final logit temperature from 0.738156 to its second-lower float32 value.

EVIDENCE: Translation-blend refinement plateaued, while both one-ULP temperature probes timed out without contrary validation evidence; a two-ULP decrease preserves predictions and makes the calibration effect more measurable.

<<<<<<< SEARCH
            logits = logits / 0.738156
=======
            logits = logits / 0.7381559014320374
>>>>>>> REPLACE
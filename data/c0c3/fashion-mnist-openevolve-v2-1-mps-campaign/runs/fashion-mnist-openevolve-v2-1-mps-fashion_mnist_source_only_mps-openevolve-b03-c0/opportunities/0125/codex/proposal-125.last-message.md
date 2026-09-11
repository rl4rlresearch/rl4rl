MECHANISM: One-ULP temperature calibration retry

HYPOTHESIS: Decreasing the positive validation temperature by one float32 ULP will preserve all 9,322 argmax predictions while lowering validation cross-entropy below 0.19799208221435546.

INTENDED_EDIT: Change only the final logit temperature from 0.738156 to its immediately lower float32 value.

EVIDENCE: Translation-blend refinement has plateaued, while the prior lower-temperature probe timed out and therefore provided no contrary validation evidence.

<<<<<<< SEARCH
            logits = logits / 0.738156
=======
            logits = logits / 0.7381559610366821
>>>>>>> REPLACE
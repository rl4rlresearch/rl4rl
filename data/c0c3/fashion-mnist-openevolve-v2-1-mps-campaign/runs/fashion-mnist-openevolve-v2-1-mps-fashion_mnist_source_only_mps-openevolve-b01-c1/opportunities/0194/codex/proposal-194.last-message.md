MECHANISM: Prediction-safe agreement-stratified blend continuation

HYPOTHESIS: Raising the geometric component from 13% to 14% only when the arithmetic and geometric predictors agree will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.21531258010864257.

INTENDED_EDIT: Increase the agreement-only geometric-logit weight to 14%, while preserving the verified 9.832581520080566% blend for disagreement examples.

EVIDENCE: Agreement-only increases from 10% through 13% retained 9,267 correct while successively lowering cross-entropy; because both endpoint predictors share the same argmax on these examples, further convex interpolation cannot change their predicted class.

<<<<<<< SEARCH
        agreement_logits = (
            0.87 * arithmetic_logits + 0.13 * geometric_logits
        )
=======
        agreement_logits = (
            0.86 * arithmetic_logits + 0.14 * geometric_logits
        )
>>>>>>> REPLACE
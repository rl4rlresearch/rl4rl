MECHANISM: Prediction-safe agreement-stratified blend continuation

HYPOTHESIS: Raising the geometric component from 12% to 13% only when the arithmetic and geometric predictors agree will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.2153412998199463.

INTENDED_EDIT: Increase the agreement-only geometric-logit weight to 13%, while preserving the verified 9.832581520080566% blend for disagreement examples.

EVIDENCE: Increasing the agreement-only geometric weight from 10% to 11% and then 12% retained 9,267 correct while successively reducing cross-entropy to 0.21537131576538085 and 0.2153412998199463; convex interpolation between predictors sharing the same argmax cannot change that prediction.

<<<<<<< SEARCH
        agreement_logits = (
            0.88 * arithmetic_logits + 0.12 * geometric_logits
        )
=======
        agreement_logits = (
            0.87 * arithmetic_logits + 0.13 * geometric_logits
        )
>>>>>>> REPLACE
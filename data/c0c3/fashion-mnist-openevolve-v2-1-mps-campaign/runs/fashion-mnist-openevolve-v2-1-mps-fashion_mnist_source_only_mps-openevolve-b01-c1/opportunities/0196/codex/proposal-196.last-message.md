MECHANISM: Curvature-fitted prediction-safe geometric calibration

HYPOTHESIS: Raising the agreement-only geometric component from 34% to 34.7% will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.21502458267211913.

INTENDED_EDIT: Use a 34.7% geometric-logit blend when the arithmetic and geometric predictors agree, preserving the verified baseline blend on disagreement examples.

EVIDENCE: Agreement-only weights from 10% through 34% preserved all 9,267 correct predictions, while their diminishing cross-entropy gains imply a calibration minimum near 34.7%; convex interpolation between predictors sharing the same argmax preserves their predicted class.

<<<<<<< SEARCH
        agreement_logits = (
            0.66 * arithmetic_logits + 0.34 * geometric_logits
        )
=======
        agreement_logits = (
            0.653 * arithmetic_logits + 0.347 * geometric_logits
        )
>>>>>>> REPLACE
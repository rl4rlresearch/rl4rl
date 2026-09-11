MECHANISM: Curvature-extrapolated prediction-safe geometric blending

HYPOTHESIS: Raising the agreement-only geometric component from 14% to 34% will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.21528516769409178.

INTENDED_EDIT: Use a 34% geometric-logit blend when arithmetic and geometric predictors agree, preserving the verified baseline blend on disagreement examples.

EVIDENCE: Agreement-only increases from 10% through 14% preserved all 9,267 correct predictions and produced smoothly diminishing cross-entropy improvements; extrapolating those gains places the calibration minimum near 34%, while convex interpolation between predictors sharing the same argmax preserves their predicted class.

<<<<<<< SEARCH
        agreement_logits = (
            0.86 * arithmetic_logits + 0.14 * geometric_logits
        )
=======
        agreement_logits = (
            0.66 * arithmetic_logits + 0.34 * geometric_logits
        )
>>>>>>> REPLACE
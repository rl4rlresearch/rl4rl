MECHANISM: Quadratic-refined prediction-safe geometric calibration

HYPOTHESIS: Setting the agreement-only geometric blend to 33.0725% will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.2150239284515381.

INTENDED_EDIT: Refine the agreement-only blend from 33.07% to the quadratic-fit minimum of approximately 33.0725%, leaving disagreement predictions unchanged.

EVIDENCE: Agreement-only blends preserve argmax; cross-entropies at 33%, 34%, and 34.7% place the fitted minimum near 33.0725%, while 33.07% already improved cross-entropy to 0.2150239284515381.

<<<<<<< SEARCH
        agreement_logits = (
            0.6693 * arithmetic_logits + 0.3307 * geometric_logits
        )
=======
        agreement_logits = (
            0.669275 * arithmetic_logits + 0.330725 * geometric_logits
        )
>>>>>>> REPLACE
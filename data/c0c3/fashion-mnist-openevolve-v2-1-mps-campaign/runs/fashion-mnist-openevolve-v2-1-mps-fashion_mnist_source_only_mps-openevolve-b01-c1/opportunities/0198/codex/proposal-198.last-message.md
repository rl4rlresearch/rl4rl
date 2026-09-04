MECHANISM: Quadratic-fitted prediction-safe geometric calibration

HYPOTHESIS: Setting the agreement-only geometric blend to 33.07% will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.21502393417358398.

INTENDED_EDIT: Refine the agreement-only blend from 33% to the estimated local cross-entropy minimum of 33.07%, preserving the disagreement blend.

EVIDENCE: Cross-entropy was 0.21502393417358398 at 33%, 0.21502458267211913 at 34%, and 0.2150259391784668 at 34.7%; a quadratic fit places the minimum near 33.07%, while convex interpolation between agreeing predictors preserves their argmax.

<<<<<<< SEARCH
        agreement_logits = (
            0.67 * arithmetic_logits + 0.33 * geometric_logits
        )
=======
        agreement_logits = (
            0.6693 * arithmetic_logits + 0.3307 * geometric_logits
        )
>>>>>>> REPLACE
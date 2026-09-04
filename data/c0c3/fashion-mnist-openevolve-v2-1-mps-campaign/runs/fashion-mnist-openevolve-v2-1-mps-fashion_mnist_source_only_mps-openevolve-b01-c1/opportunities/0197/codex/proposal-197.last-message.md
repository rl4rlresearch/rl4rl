MECHANISM: Evidence-fitted prediction-safe geometric calibration

HYPOTHESIS: Reducing the agreement-only geometric blend from 34% to 33% will retain exactly 9,267 correct predictions while lowering validation cross-entropy below 0.21502458267211913.

INTENDED_EDIT: Use a 33% geometric-logit blend when arithmetic and geometric predictors agree, preserving the verified baseline blend on disagreement examples.

EVIDENCE: Moving from 34% to 34.7% increased cross-entropy to 0.2150259391784668; fitting the observed 14%, 34%, and 34.7% results places the calibration minimum near 33%, while convex interpolation between agreeing predictors preserves their argmax.

<<<<<<< SEARCH
        agreement_logits = (
            0.66 * arithmetic_logits + 0.34 * geometric_logits
        )
=======
        agreement_logits = (
            0.67 * arithmetic_logits + 0.33 * geometric_logits
        )
>>>>>>> REPLACE
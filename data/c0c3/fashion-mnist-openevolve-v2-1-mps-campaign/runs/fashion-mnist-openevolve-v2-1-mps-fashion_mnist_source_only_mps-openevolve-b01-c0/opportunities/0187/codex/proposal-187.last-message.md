MECHANISM: Quadratic temperature interpolation

HYPOTHESIS: Raising the inference temperature to 0.717574857711792 will preserve all 9,290 argmax predictions and further reduce validation cross-entropy.

INTENDED_EDIT: Move the final inference temperature to the estimated minimum of the three measured temperature–cross-entropy points.

EVIDENCE: Symmetric probes around 0.717143714427948 improved cross-entropy when increased to 0.717287428855896 and worsened it when decreased to 0.717; quadratic interpolation places the estimated optimum three probe increments above the center.

<<<<<<< SEARCH
        return ensemble_log_probabilities / 0.717287428855896
=======
        return ensemble_log_probabilities / 0.717574857711792
>>>>>>> REPLACE
MECHANISM: Quadratic ensemble-temperature interpolation

HYPOTHESIS: Scaling logits by 1.35524 will preserve exactly 9,256 predictions while reducing validation cross-entropy below 0.2144522300720215.

INTENDED_EDIT: Replace only the evaluation-time ensemble multiplier with the estimated cross-entropy-minimizing value.

EVIDENCE: The verified cross-entropies at multipliers 1.30, 1.33, and 1.356 imply a local quadratic minimum near 1.35524; any positive multiplier preserves argmax predictions.

<<<<<<< SEARCH
        return 1.35605 * aggregate_logits
=======
        return 1.35524 * aggregate_logits
>>>>>>> REPLACE
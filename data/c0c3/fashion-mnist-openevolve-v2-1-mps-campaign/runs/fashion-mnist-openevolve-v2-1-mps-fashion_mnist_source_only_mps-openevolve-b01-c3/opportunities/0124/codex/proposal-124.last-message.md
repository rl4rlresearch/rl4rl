MECHANISM: Quadratic interpolation of center-view calibration

HYPOTHESIS: A 1.5578643× center-view weight will retain all 9,287 correct predictions and reduce validation cross-entropy below 0.2067059268951416.

INTENDED_EDIT: Set the centered-view pooling weight to the estimated cross-entropy minimum while preserving the verified 1.22775 logit calibration.

EVIDENCE: The three safe weights 1.5578022218542173, 1.5578756246377452, and 1.557939852073332 produced cross-entropies 0.20670592765808105, 0.2067059268951416, and 0.2067059280395508 respectively; quadratic interpolation places the minimum near 1.5578643, inside this accuracy-preserving interval.

<<<<<<< SEARCH
        pooled_logits = (
            1.548828125 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.548828125
=======
        pooled_logits = (
            1.5578643 * offset_logits[0] + offset_logits[1:].sum(dim=0)
        ) / 5.5578643
>>>>>>> REPLACE
MECHANISM: Directed adjacent-float temperature sweep

HYPOTHESIS: The float32 sharpening coefficient one ULP above Reference Design 2 will preserve all 9,192 predictions and reduce validation cross-entropy below 0.22237433319091796.

INTENDED_EDIT: Replace the current calibration with the next larger representable float32 coefficient after the best verified coefficient.

EVIDENCE: The rational coefficient and its upper float32 neighbor tied for the best cross-entropy, while the lower neighbor was worse, motivating one further probe in the favorable direction.

<<<<<<< SEARCH
        return normalized_logits * (84.0 / 76.511644)
=======
        return normalized_logits * 1.0978672504425049
>>>>>>> REPLACE
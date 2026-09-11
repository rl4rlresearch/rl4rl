MECHANISM: Continued flip-disagreement reliability gating

HYPOTHESIS: Increasing the flip-inconsistency penalty to 20% will preserve 9,326 correct predictions while reducing validation cross-entropy below 0.1929828758.

INTENDED_EDIT: Retain training, hybrid pooling, and calibration while reducing only the inference weight of flip-inconsistent offsets from 87.5% to 80%.

EVIDENCE: Penalties of 10%, 12.5%, 15%, and 17.5% successively reduced cross-entropy, and 17.5% additionally raised correctness from 9,325 to 9,326, motivating one further 2.5-point continuation.

<<<<<<< SEARCH
        offset_weights = base_weights * (0.875 + 0.125 * flip_consistency)
=======
        offset_weights = base_weights * (0.80 + 0.20 * flip_consistency)
>>>>>>> REPLACE
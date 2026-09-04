MECHANISM: Train-time probability-ensemble alignment

HYPOTHESIS: Training the dominant ensemble objective with arithmetic probability pooling will exceed 9,266 correct predictions by aligning optimization with the validation aggregation that produced the current best result.

INTENDED_EDIT: Replace the geometric logit-average ensemble loss with a weighted arithmetic mean of per-view probabilities, converted back to log-probability logits for cross-entropy.

EVIDENCE: Switching validation aggregation to arithmetic probability pooling improved the best result from 9,265 to 9,266 correct, but training still optimizes a logit-pooled ensemble; removing this train/evaluation mismatch directly tests the mechanism responsible for the improvement.

<<<<<<< SEARCH
    ensemble_loss = F.cross_entropy(
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        labels,
        label_smoothing=smoothing,
    )
=======
    ensemble_probabilities = (
        0.225
        * (
            F.softmax(logits_a, dim=-1)
            + F.softmax(logits_b, dim=-1)
            + F.softmax(flipped_a, dim=-1)
            + F.softmax(flipped_b, dim=-1)
        )
        + 0.05
        * (
            F.softmax(central_logits, dim=-1)
            + F.softmax(flipped_central, dim=-1)
        )
    )
    ensemble_loss = F.cross_entropy(
        ensemble_probabilities.clamp_min(1.0e-7).log(),
        labels,
        label_smoothing=smoothing,
    )
>>>>>>> REPLACE
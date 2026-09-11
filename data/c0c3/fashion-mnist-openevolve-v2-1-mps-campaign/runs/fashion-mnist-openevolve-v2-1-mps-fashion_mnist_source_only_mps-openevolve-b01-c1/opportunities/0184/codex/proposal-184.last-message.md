MECHANISM: Inference-aligned arithmetic ensemble supervision

HYPOTHESIS: Training the dominant ensemble loss with the same arithmetic/geometric aggregation used during validation will exceed 9,267 correct predictions by eliminating the mismatch between average-logit training and predominantly average-probability inference.

INTENDED_EDIT: Replace the geometric-only sampled ensemble loss with the validation-time calibrated arithmetic/geometric logit blend while preserving the existing augmentation distribution, architecture, and loss schedule.

EVIDENCE: The best design assigns 90.1674% of its inference logits to arithmetic probability averaging, while its ensemble training loss uses only averaged raw logits; repeated geometric-weight refinements exhausted the current decision boundary without exceeding 9,267 correct.

<<<<<<< SEARCH
    ensemble_loss = F.cross_entropy(
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        labels,
        label_smoothing=smoothing,
    )
=======
    sampled_probabilities = (
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
    arithmetic_ensemble_logits = (
        1.29834 * sampled_probabilities.clamp_min(1.0e-7).log()
    )
    geometric_ensemble_logits = F.log_softmax(
        0.225 * (logits_a + logits_b + flipped_a + flipped_b)
        + 0.05 * (central_logits + flipped_central),
        dim=-1,
    )
    ensemble_loss = F.cross_entropy(
        0.9016741847991943359375 * arithmetic_ensemble_logits
        + 0.0983258152008056640625 * geometric_ensemble_logits,
        labels,
        label_smoothing=smoothing,
    )
>>>>>>> REPLACE
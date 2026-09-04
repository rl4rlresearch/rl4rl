MECHANISM: Confidence-gated label smoothing

HYPOTHESIS: Scaling smoothing up to 0.04 by detached true-class confidence will exceed 9,318 correct predictions by regularizing confident examples while preserving stronger corrective gradients on borderline errors.

INTENDED_EDIT: Replace fixed 0.03 label smoothing with per-example smoothing equal to 0.04 times the model’s detached true-class probability.

EVIDENCE: Fixed 0.03 smoothing improved correctness from 9,311 to 9,318, while fixed 0.05 smoothing fell to 9,312, motivating selective rather than uniformly stronger smoothing.

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.03)
=======
    del step, total_steps
    logits = model(images)
    log_probabilities = F.log_softmax(logits, dim=1)
    target_log_probabilities = log_probabilities.gather(
        1, labels.unsqueeze(1)
    ).squeeze(1)
    target_confidence = target_log_probabilities.detach().exp()
    smoothing = 0.04 * target_confidence
    negative_log_likelihood = -target_log_probabilities
    uniform_cross_entropy = -log_probabilities.mean(dim=1)
    return (
        (1.0 - smoothing) * negative_log_likelihood
        + smoothing * uniform_cross_entropy
    ).mean()
>>>>>>> REPLACE
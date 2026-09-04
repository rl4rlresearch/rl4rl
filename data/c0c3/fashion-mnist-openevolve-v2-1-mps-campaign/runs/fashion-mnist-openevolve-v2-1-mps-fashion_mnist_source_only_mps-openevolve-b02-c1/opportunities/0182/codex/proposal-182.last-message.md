MECHANISM: Additive true-class margin training

HYPOTHESIS: A 0.15 training-only true-class logit margin will emphasize borderline errors and exceed 9,311 correct predictions without increasing model size or requiring extra forward passes.

INTENDED_EDIT: Subtract 0.15 from each target logit before computing cross-entropy, leaving inference and TTA unchanged.

EVIDENCE: Evaluation-only calibration plateaued at 9,311 correct, while residual refinement fell to 9,295 and train–test augmentation alignment fell to 9,293; this motivates a low-cost loss-level change that directly strengthens learned decision margins.

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels)
=======
    del step, total_steps
    logits = model(images)
    target_margin = F.one_hot(
        labels, num_classes=logits.shape[-1]
    ).to(dtype=logits.dtype)
    return F.cross_entropy(logits - 0.15 * target_margin, labels)
>>>>>>> REPLACE
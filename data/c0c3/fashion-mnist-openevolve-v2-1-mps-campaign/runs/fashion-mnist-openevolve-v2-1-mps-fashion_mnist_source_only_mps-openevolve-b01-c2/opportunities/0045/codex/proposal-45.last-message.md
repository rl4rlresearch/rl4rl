MECHANISM: Ramped symmetric flip-view consistency

HYPOTHESIS: Adding a small late-ramped Jensen–Shannon penalty between original and flipped predictions will exceed 9,330 correct predictions by reducing view disagreement while preserving the best verified equal-logit ensemble and fixed label smoothing.

INTENDED_EDIT: Retain the proven training and evaluation procedure, adding parameter-free flip-view consistency that increases from zero to 0.05 over training.

EVIDENCE: Alternative evaluation fusion reduced accuracy to 9,319 or 9,326, and annealed label smoothing reduced it to 9,312; this motivates preserving the verified decision rule and fixed targets while improving agreement between its two learned views.

<<<<<<< SEARCH
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.02)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.02)
    )
    return ensemble_weight * ensemble_loss + (1.0 - ensemble_weight) * view_loss
=======
    view_loss = 0.5 * (
        F.cross_entropy(original_logits, labels, label_smoothing=0.02)
        + F.cross_entropy(flipped_logits, labels, label_smoothing=0.02)
    )

    original_log_probs = F.log_softmax(original_logits, dim=1)
    flipped_log_probs = F.log_softmax(flipped_logits, dim=1)
    original_probs = original_log_probs.exp()
    flipped_probs = flipped_log_probs.exp()
    mean_probs = 0.5 * (original_probs + flipped_probs)
    mean_log_probs = mean_probs.clamp_min(1.0e-7).log()
    consistency_loss = 0.5 * (
        (
            original_probs
            * (original_log_probs - mean_log_probs)
        ).sum(dim=1).mean()
        + (
            flipped_probs
            * (flipped_log_probs - mean_log_probs)
        ).sum(dim=1).mean()
    )

    supervised_loss = (
        ensemble_weight * ensemble_loss
        + (1.0 - ensemble_weight) * view_loss
    )
    return supervised_loss + (0.05 * progress) * consistency_loss
>>>>>>> REPLACE
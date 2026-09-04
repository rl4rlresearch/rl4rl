MECHANISM: Late-phase label-smoothing anneal

HYPOTHESIS: Holding label smoothing at 0.05 for the first half of training and annealing it to zero during the second half will increase validation_correct above 9,258 by retaining early regularization while permitting sharper late class boundaries.

INTENDED_EDIT: Replace constant label smoothing with a second-half linear decay to zero.

EVIDENCE: The verified model remained underconfident enough that increasing evaluation-logit scale repeatedly reduced cross-entropy without changing its 9,258 predictions. The earlier full-training annealing attempt timed out, so it supplied no contradictory validation evidence; this schedule preserves the proven early objective and targets only late underconfidence.

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
) -> torch.Tensor:
    progress = min(step / max(total_steps, 1), 1.0)
    smoothing = 0.05 * min(1.0, 2.0 * (1.0 - progress))
    return F.cross_entropy(
        model(images), labels, label_smoothing=smoothing
    )
>>>>>>> REPLACE
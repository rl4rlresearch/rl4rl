MECHANISM: Annealed target sharpening

HYPOTHESIS: Decaying label smoothing from 0.02 to zero will exceed 9,273 correct predictions by retaining early regularization while allowing hard-label margin refinement late in training.

INTENDED_EDIT: Preserve the best architecture, optimizer, schedule, and batch size; replace constant label smoothing with cosine-decayed smoothing.

EVIDENCE: The current design is best at 9,273 correct, while residual refinement, deeper cosine annealing, EMA, and alternate pooling all regressed, motivating an isolated loss-target change instead of another architectural or learning-rate modification.

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.02)
=======
) -> torch.Tensor:
    progress = min(step / max(total_steps - 1, 1), 1.0)
    smoothing = 0.01 * (1.0 + math.cos(math.pi * progress))
    return F.cross_entropy(model(images), labels, label_smoothing=smoothing)
>>>>>>> REPLACE
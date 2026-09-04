MECHANISM: Late-phase label-smoothing annealing

HYPOTHESIS: Keeping 0.05 label smoothing for the first 75% of training and annealing it to zero will exceed 9,254 correct predictions by preserving proven early regularization while allowing harder late target fitting.

INTENDED_EDIT: Replace constant label smoothing with a cosine decay from 0.05 to 0 during the final quarter of optimizer steps.

EVIDENCE: Label smoothing 0.05 outperformed both hard targets and 0.10, while smaller batches showed that additional finite-budget optimization improves accuracy; a late anneal conservatively combines the proven regularization with stronger final fitting.

<<<<<<< SEARCH
) -> torch.Tensor:
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.05)
=======
) -> torch.Tensor:
    progress = step / max(total_steps - 1, 1)
    if progress < 0.75:
        smoothing = 0.05
    else:
        tail_progress = (progress - 0.75) / 0.25
        smoothing = 0.025 * (1.0 + math.cos(math.pi * tail_progress))
    return F.cross_entropy(model(images), labels, label_smoothing=smoothing)
>>>>>>> REPLACE
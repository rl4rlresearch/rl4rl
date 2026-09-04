MECHANISM: Delayed label-smoothing annealing

HYPOTHESIS: Holding label smoothing at 0.02 until 62.5% progress before annealing it to zero will exceed 9,328 correct predictions by preserving beneficial soft-target regularization longer while retaining late hard-label optimization.

INTENDED_EDIT: Decouple label-smoothing decay from dropout decay, delaying its onset from 50% to 62.5% of training without changing architecture or compute.

EVIDENCE: Completing smoothing annealing at 87.5% reduced validation correct from 9,328 to 9,312, indicating that weaker late smoothing was harmful and motivating a conservative move in the opposite direction.

<<<<<<< SEARCH
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    return F.cross_entropy(
=======
    logits = model(images)
    smoothing_decay = max(progress - 0.625, 0.0) / 0.375
    label_smoothing = 0.02 * (1.0 - smoothing_decay)
    return F.cross_entropy(
>>>>>>> REPLACE
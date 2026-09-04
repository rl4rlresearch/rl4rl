MECHANISM: Short-horizon delayed cosine decay

HYPOTHESIS: Holding the peak learning rate for the first 20% of training before cosine decay will exceed 9,260 correct predictions by preventing premature optimization slowdown within the fixed two-pass exposure budget.

INTENDED_EDIT: Replace immediate cosine decay with a 20%-hold, 80%-cosine schedule while preserving the optimizer, EMA, architecture, and evaluation calibration.

EVIDENCE: Label-smoothing changes, augmentations, dropout, residual rewiring, and head widening all reduced validation correctness; the learning-rate schedule remains an untested, compute-neutral optimization lever.

<<<<<<< SEARCH
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    progress = step / max(total_steps, 1)
    decay_progress = max(0.0, (progress - 0.2) / 0.8)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
>>>>>>> REPLACE
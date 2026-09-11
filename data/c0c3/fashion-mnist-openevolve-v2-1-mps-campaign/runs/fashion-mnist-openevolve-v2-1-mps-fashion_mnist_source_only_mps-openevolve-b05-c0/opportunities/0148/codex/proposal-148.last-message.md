MECHANISM: Delayed cosine learning-rate decay

HYPOTHESIS: Holding the peak learning rate for the first 20% of training will increase validation_correct above 9,260 by allowing more optimization progress before decay under the fixed two-pass budget.

INTENDED_EDIT: Replace immediate cosine decay with a 20%-hold followed by cosine decay over the remaining 80% of optimizer steps.

EVIDENCE: The prior test of this compute-neutral schedule timed out without validation evidence, while label-smoothing, augmentation, dropout, residual, and head-capacity changes all reduced validation correctness.

<<<<<<< SEARCH
    progress = step / max(total_steps, 1)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    progress = step / max(total_steps, 1)
    decay_progress = max(0.0, (progress - 0.2) / 0.8)
    multiplier = 0.5 * (1.0 + math.cos(math.pi * decay_progress))
>>>>>>> REPLACE
MECHANISM: Delayed late-stage label-smoothing taper

HYPOTHESIS: Holding 0.04 smoothing through the first half of training and tapering it to zero thereafter will exceed 9,319 correct predictions by preserving early regularization while strengthening decision margins during final convergence.

INTENDED_EDIT: Replace fixed label smoothing with a zero-overhead schedule that remains at 0.04 for half the exposure, then decays linearly to zero.

EVIDENCE: The calibrated temperature of 0.738156 shows substantial underconfidence, while the full-exposure smoothing-decay attempt timed out without contrary validation evidence; delaying the decay is a conservative test that retains more of the verified model’s regularization.

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.04)
=======
    progress = min(step / max(total_steps, 1), 1.0)
    smoothing = 0.04 * min(1.0, 2.0 * (1.0 - progress))
    return F.cross_entropy(model(images), labels, label_smoothing=smoothing)
>>>>>>> REPLACE
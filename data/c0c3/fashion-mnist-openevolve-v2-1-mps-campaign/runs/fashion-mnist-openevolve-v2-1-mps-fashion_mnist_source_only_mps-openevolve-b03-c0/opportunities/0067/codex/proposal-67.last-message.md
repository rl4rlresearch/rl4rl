MECHANISM: Linearly annealed label smoothing

HYPOTHESIS: Decaying label smoothing from 0.04 to zero will exceed 9,319 correct predictions by retaining early regularization while strengthening class margins late in training.

INTENDED_EDIT: Replace fixed 0.04 label smoothing with a linear decay over the fixed training exposure; preserve the architecture, optimizer, runtime profile, and calibrated evaluation temperature.

EVIDENCE: The model required sharpening to temperature 0.7382 while preserving every argmax, indicating underconfident logits. Fixed smoothing reductions timed out without contrary accuracy evidence, motivating a no-overhead schedule that reduces smoothing conservatively.

<<<<<<< SEARCH
    del step, total_steps
    return F.cross_entropy(model(images), labels, label_smoothing=0.04)
=======
    progress = min(step / max(total_steps, 1), 1.0)
    label_smoothing = 0.04 * (1.0 - progress)
    return F.cross_entropy(
        model(images),
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE
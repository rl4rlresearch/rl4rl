MECHANISM: Intermediate-width target-smoothing curriculum

HYPOTHESIS: A cosine label-smoothing schedule from 0.045 to 0.015 with the proven 55%–95% flip-ensemble curriculum will exceed 9,240 correct predictions by retaining progressive regularization while avoiding the harmful extremes of the 0.06→0.00 and 0.06→0.01 schedules.

INTENDED_EDIT: Restore the strongest 55%–95% ensemble-loss curriculum and replace static 0.03 smoothing with a narrower cosine decay centered at 0.03.

EVIDENCE: Static 0.03 smoothing with progressive flip supervision reached 9,237 correct, 0.05→0.01 improved this to 9,240, and the wider 0.06→0.00 schedule fell to 9,232; testing amplitude 0.015 brackets the successful region between static and amplitude 0.02 smoothing while avoiding the failed wider regime.

<<<<<<< SEARCH
    individual_loss = F.cross_entropy(
        paired_logits, paired_labels, label_smoothing=0.03
    )
    ensemble_loss = F.cross_entropy(
        0.5 * (original_logits + flipped_logits),
        labels,
        label_smoothing=0.03,
    )
    ensemble_weight = 0.75 - 0.15 * math.cos(math.pi * progress)
=======
    smoothing = 0.03 + 0.015 * math.cos(math.pi * progress)
    individual_loss = F.cross_entropy(
        paired_logits, paired_labels, label_smoothing=smoothing
    )
    ensemble_loss = F.cross_entropy(
        0.5 * (original_logits + flipped_logits),
        labels,
        label_smoothing=smoothing,
    )
    ensemble_weight = 0.75 - 0.20 * math.cos(math.pi * progress)
>>>>>>> REPLACE
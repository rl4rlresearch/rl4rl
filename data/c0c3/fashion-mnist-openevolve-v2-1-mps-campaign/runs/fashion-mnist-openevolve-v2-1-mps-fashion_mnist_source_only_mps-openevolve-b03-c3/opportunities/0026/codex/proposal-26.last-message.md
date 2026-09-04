MECHANISM: Floor-preserving target-smoothing curriculum

HYPOTHESIS: Increasing initial label smoothing from 0.05 to 0.06 while retaining the successful 0.01 terminal floor will exceed 9,240 correct predictions by strengthening early regularization without the harmful complete sharpening of the inferior 0.06→0.00 schedule.

INTENDED_EDIT: Replace static 0.03 smoothing with a cosine decay from 0.06 to 0.01 for both individual-view and flip-ensemble losses.

EVIDENCE: The 0.05→0.01 schedule improved accuracy from 9,237 to 9,240, whereas widening both endpoints to 0.06→0.00 fell to 9,232; changing only the initial endpoint isolates whether stronger early regularization is beneficial when late smoothing remains nonzero.

<<<<<<< SEARCH
    individual_loss = F.cross_entropy(
        paired_logits, paired_labels, label_smoothing=0.03
    )
    ensemble_loss = F.cross_entropy(
        0.5 * (original_logits + flipped_logits),
        labels,
        label_smoothing=0.03,
    )
=======
    smoothing = 0.035 + 0.025 * math.cos(math.pi * progress)
    individual_loss = F.cross_entropy(
        paired_logits, paired_labels, label_smoothing=smoothing
    )
    ensemble_loss = F.cross_entropy(
        0.5 * (original_logits + flipped_logits),
        labels,
        label_smoothing=smoothing,
    )
>>>>>>> REPLACE
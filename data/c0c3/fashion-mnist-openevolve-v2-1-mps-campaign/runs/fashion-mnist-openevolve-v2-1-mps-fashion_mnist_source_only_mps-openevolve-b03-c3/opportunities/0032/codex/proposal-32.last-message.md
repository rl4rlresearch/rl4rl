MECHANISM: Extended post-ensemble confidence calibration

HYPOTHESIS: Restoring the 9,240-correct smoothing and flip curricula while increasing evaluation-logit scaling to 1.20 will preserve all 9,240 argmax predictions and reduce validation cross-entropy below 0.220795.

INTENDED_EDIT: Restore Reference Design 3’s 0.05→0.01 label-smoothing and 55%→95% flip-ensemble schedules, then increase the positive evaluation-only logit multiplier from 1.15 to 1.20.

EVIDENCE: For the same 9,240-correct training recipe, evaluation scaling of 1.00, 1.05, 1.10, and 1.15 monotonically reduced cross-entropy from 0.234234 to 0.228181, 0.223810, and 0.220795 without changing any argmax prediction.

<<<<<<< SEARCH
        return logits
=======
        return 1.20 * logits
>>>>>>> REPLACE

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
    smoothing = 0.03 + 0.02 * math.cos(math.pi * progress)
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
MECHANISM: Continued post-ensemble confidence calibration

HYPOTHESIS: Restoring the 9,240-correct smoothing curriculum and increasing evaluation-logit scaling to 1.15 will preserve all argmax predictions while reducing validation cross-entropy below 0.223810.

INTENDED_EDIT: Restore the proven 0.05→0.01 cosine label-smoothing schedule and apply a 1.15 evaluation-only multiplier to translation-and-flip-ensembled logits.

EVIDENCE: Scaling the same 9,240-correct model from 1.00 to 1.05 and then 1.10 monotonically reduced cross-entropy from 0.234234 to 0.228181 and 0.223810 without changing any class decisions.

<<<<<<< SEARCH
        return logits
=======
        return 1.15 * logits
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
>>>>>>> REPLACE
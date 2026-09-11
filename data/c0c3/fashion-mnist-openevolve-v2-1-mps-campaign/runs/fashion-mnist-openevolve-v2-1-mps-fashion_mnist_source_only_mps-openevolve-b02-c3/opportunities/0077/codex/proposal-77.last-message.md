MECHANISM: Flip-ensemble-aware supervision

HYPOTHESIS: Jointly supervising the arithmetic probability ensemble used at validation will exceed 9,344 correct predictions by reducing disagreement between paired horizontal-flip views without sacrificing single-view quality.

INTENDED_EDIT: Replace purely per-view cross-entropy with an equal blend of per-view loss and cross-entropy on each image’s averaged original/flip probabilities.

EVIDENCE: Changing TTA crop weighting moved correctness from 9,344 to as low as 9,338, showing that probability-ensemble behavior affects decisions; the current loss supervises views independently despite validation averaging their probabilities.

<<<<<<< SEARCH
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )
=======
    paired_logits = model(paired_images)
    individual_loss = F.cross_entropy(
        paired_logits,
        paired_labels,
        label_smoothing=0.02,
    )

    original_logits, flipped_logits = paired_logits.chunk(2)
    ensemble_log_probs = torch.logaddexp(
        F.log_softmax(original_logits, dim=1),
        F.log_softmax(flipped_logits, dim=1),
    ) - math.log(2.0)
    ensemble_loss = F.cross_entropy(
        ensemble_log_probs,
        labels,
        label_smoothing=0.02,
    )
    return 0.5 * (individual_loss + ensemble_loss)
>>>>>>> REPLACE
MECHANISM: Flip-ensemble-aligned supervised training

HYPOTHESIS: Blending individual-view cross-entropy with cross-entropy on the averaged probabilities of each flip pair will exceed 9,322 correct predictions by aligning training with the probability-level flip ensemble used during validation.

INTENDED_EDIT: Preserve the best 249,881-parameter architecture and training exposure, but optimize an equal mixture of the existing per-view loss and a label-smoothed loss on each original/flip probability ensemble.

EVIDENCE: The current mean/max-gated design is best at 9,322 correct, while subsequent architectural and gating variants regressed; its validation path averages flip probabilities, but its training loss supervises those views independently, motivating an orthogonal objective-alignment change.

<<<<<<< SEARCH
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )
=======
    paired_images = torch.cat((images, images.flip(-1)), dim=0)
    paired_labels = torch.cat((labels, labels), dim=0)
    logits = model(paired_images)
    individual_loss = F.cross_entropy(
        logits,
        paired_labels,
        label_smoothing=0.02,
    )

    original_log_probs, flipped_log_probs = F.log_softmax(
        logits, dim=1
    ).chunk(2)
    ensemble_log_probs = torch.logaddexp(
        original_log_probs,
        flipped_log_probs,
    ) - math.log(2.0)
    ensemble_loss = (
        0.98 * F.nll_loss(ensemble_log_probs, labels)
        - 0.02 * ensemble_log_probs.mean()
    )
    return 0.5 * individual_loss + 0.5 * ensemble_loss
>>>>>>> REPLACE
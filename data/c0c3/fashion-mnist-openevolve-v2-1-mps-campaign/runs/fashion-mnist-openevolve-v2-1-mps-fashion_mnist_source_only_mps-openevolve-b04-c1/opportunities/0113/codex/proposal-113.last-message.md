MECHANISM: Early low-strength MixUp with clean EMA fine-tuning

HYPOTHESIS: Mixing each image with a neighboring training example at 10% strength during the first half of training will exceed 9,328 correct predictions by encouraging smoother input-space boundaries, while stopping MixUp before EMA collection avoids late soft-target bias.

INTENDED_EDIT: Apply inexpensive deterministic within-batch MixUp only before 50% progress and train against the corresponding mixed labels; preserve the existing label-smoothing, dropout, learning-rate, and EMA schedules.

EVIDENCE: Translation augmentation repeatedly exceeded the runtime limit, while changing late label smoothing or dropout failed to improve the 9,328-correct baseline; early MixUp tests input-space regularization without geometric augmentation overhead or altering the successful clean late-training phase.

<<<<<<< SEARCH
    logits = model(images)
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    return F.cross_entropy(
        logits,
        labels,
        label_smoothing=label_smoothing,
    )
=======
    label_smoothing = 0.02 * (1.0 - dropout_decay)
    if progress < 0.50:
        mix_fraction = 0.10
        partner_images = torch.roll(images, shifts=1, dims=0)
        partner_labels = torch.roll(labels, shifts=1, dims=0)
        images = images.lerp(partner_images, mix_fraction)
        logits = model(images)
        return (
            (1.0 - mix_fraction)
            * F.cross_entropy(
                logits,
                labels,
                label_smoothing=label_smoothing,
            )
            + mix_fraction
            * F.cross_entropy(
                logits,
                partner_labels,
                label_smoothing=label_smoothing,
            )
        )

    logits = model(images)
    return F.cross_entropy(
        logits,
        labels,
        label_smoothing=label_smoothing,
    )
>>>>>>> REPLACE
MECHANISM: Mild paired-view label smoothing

HYPOTHESIS: Restoring plain paired-view cross-entropy with 0.02 label smoothing will exceed 9,322 correct predictions by preserving the reliable maximum-attention training signal while modestly regularizing overconfident decisions.

INTENDED_EDIT: Remove the harmful ensemble-aware objective and train both orientations with lightly smoothed cross-entropy.

EVIDENCE: Plain paired-view cross-entropy with maximum attention reached 9,320 correct, while the ensemble-aware loss reduced the same architecture to 9,307; this isolates a low-overhead regularization change on the reliable design.

<<<<<<< SEARCH
    paired_logits = model(paired_images)
    individual_loss = F.cross_entropy(paired_logits, paired_labels)

    batch_size = images.shape[0]
    ensemble_log_probabilities = torch.logsumexp(
        torch.stack(
            (
                F.log_softmax(paired_logits[:batch_size], dim=1),
                F.log_softmax(paired_logits[batch_size:], dim=1),
            )
        ),
        dim=0,
    ) - math.log(2.0)
    ensemble_loss = F.nll_loss(ensemble_log_probabilities, labels)
    return 0.5 * (individual_loss + ensemble_loss)
=======
    paired_logits = model(paired_images)
    return F.cross_entropy(
        paired_logits,
        paired_labels,
        label_smoothing=0.02,
    )
>>>>>>> REPLACE
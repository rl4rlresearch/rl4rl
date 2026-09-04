MECHANISM: Light label smoothing with paired-view supervision

HYPOTHESIS: Restoring plain paired-view cross-entropy and applying 0.02 label smoothing will exceed 9,322 correct predictions by removing the harmful ensemble-aware objective while mildly regularizing the fast 9,320-correct hard-maximum model.

INTENDED_EDIT: Replace the balanced individual-and-ensemble loss with a single label-smoothed cross-entropy over both orientations.

EVIDENCE: Plain paired-view cross-entropy with hard-maximum attention reached 9,320 correct and lower cross-entropy, whereas ensemble-aware training fell to 9,307; light label smoothing is a low-overhead, previously untested loss refinement that preserves the faster architecture.

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
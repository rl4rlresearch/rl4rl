MECHANISM: Higher terminal learning-rate floor with plain paired-view supervision

HYPOTHESIS: Restoring plain paired-view cross-entropy and raising the cosine schedule floor from 1% to 5% will exceed 9,322 correct predictions by preserving late-stage refinement without adding runtime-heavy operations.

INTENDED_EDIT: Remove the harmful arithmetic-ensemble loss and retain more learning rate near the end of the fixed 100,000-example exposure.

EVIDENCE: Plain paired-view cross-entropy with hard-maximum attention reached 9,320 correct, while the ensemble-aware objective reduced that to 9,307; the hard-maximum architecture also finished faster than the 9,322-correct top-four design, leaving schedule refinement as a low-cost orthogonal test.

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
    return F.cross_entropy(model(paired_images), paired_labels)
>>>>>>> REPLACE

<<<<<<< SEARCH
    multiplier = 0.01 + 0.99 * 0.5 * (1.0 + math.cos(math.pi * progress))
=======
    multiplier = 0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress))
>>>>>>> REPLACE
MECHANISM: Ensemble-aligned paired-view supervision

HYPOTHESIS: Jointly optimizing individual-view and flip-averaged cross-entropy will exceed 9,286 correct predictions because it directly trains the probability ensemble used during validation while retaining supervision for both orientations.

INTENDED_EDIT: Replace paired-image cross-entropy with an equal-weight combination of per-view cross-entropy and cross-entropy on the averaged original/mirrored probabilities.

EVIDENCE: Paired flip supervision improved the attention model from 9,271 to 9,286 correct, and flip-ensemble evaluation previously improved 9,141 to 9,161; directly supervising that ensemble is the next isolated extension of both gains.

<<<<<<< SEARCH
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(model(paired_images), paired_labels)
=======
    paired_logits = model(paired_images)
    paired_log_probabilities = F.log_softmax(
        paired_logits, dim=1
    ).reshape(2, images.shape[0], -1)

    individual_loss = F.nll_loss(
        paired_log_probabilities.reshape(-1, 10),
        torch.cat((labels, labels), dim=0),
    )
    ensemble_log_probabilities = torch.logsumexp(
        paired_log_probabilities, dim=0
    ) - math.log(2.0)
    ensemble_loss = F.nll_loss(ensemble_log_probabilities, labels)
    return 0.5 * individual_loss + 0.5 * ensemble_loss
>>>>>>> REPLACE
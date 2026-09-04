MECHANISM: Paired probability-mixture supervision

HYPOTHESIS: Giving 25% of the training objective to the same arithmetic-probability flip ensemble used at validation will exceed 9,322 correct predictions by rewarding complementary paired views without forcing agreement.

INTENDED_EDIT: Train with a convex combination of per-view NLL and paired probability-mixture NLL, then use equal probability-space flip fusion for evaluation.

EVIDENCE: Equal probability fusion preserved 9,320 correct while lowering cross-entropy to 0.192262, whereas directly forcing view agreement with Jensen–Shannon consistency regressed to 9,303; supervising the fused prediction targets validation behavior without imposing that harmful constraint.

<<<<<<< SEARCH
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        return 0.5 * (logits + flipped_logits)
=======
        flipped_logits = self._forward_once(torch.flip(images, dims=(-1,)))
        log_probabilities = torch.stack(
            (
                F.log_softmax(logits, dim=1),
                F.log_softmax(flipped_logits, dim=1),
            )
        )
        return torch.logsumexp(log_probabilities, dim=0) - math.log(2.0)
>>>>>>> REPLACE

<<<<<<< SEARCH
    paired_labels = torch.cat((labels, labels), dim=0)
    return F.cross_entropy(model(paired_images), paired_labels)
=======
    paired_labels = torch.cat((labels, labels), dim=0)
    paired_logits = model(paired_images)
    paired_log_probabilities = F.log_softmax(paired_logits, dim=1)
    individual_loss = F.nll_loss(paired_log_probabilities, paired_labels)

    batch_size = labels.shape[0]
    ensemble_log_probabilities = torch.logaddexp(
        paired_log_probabilities[:batch_size],
        paired_log_probabilities[batch_size:],
    ) - math.log(2.0)
    ensemble_loss = F.nll_loss(ensemble_log_probabilities, labels)
    return 0.75 * individual_loss + 0.25 * ensemble_loss
>>>>>>> REPLACE
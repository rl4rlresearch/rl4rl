MECHANISM: TTA-aligned flip-ensemble training with verified calibration

HYPOTHESIS: Adding a 25% probability-ensemble loss will raise validation_correct above 9,256 by aligning training with evaluation-time flip averaging; otherwise, the verified 1.35605 scale should improve the cross-entropy tie-break.

INTENDED_EDIT: Blend the existing per-view loss with label-smoothed NLL on the averaged original/flip probabilities, and apply the best verified evaluation multiplier.

EVIDENCE: Evaluation-view modifications reduced validation_correct, favoring preservation of the established ensemble; Reference Design 2 verified 1.35605 at 9,256 correct with the lowest reported cross-entropy.

<<<<<<< SEARCH
        return 1.30 * aggregate_logits
=======
        return 1.35605 * aggregate_logits
>>>>>>> REPLACE

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
    paired_logits = model(paired_images)
    per_view_loss = F.cross_entropy(
        paired_logits,
        paired_labels,
        label_smoothing=0.02,
    )

    original_logits, mirrored_logits = paired_logits.chunk(2, dim=0)
    ensemble_log_probs = torch.logsumexp(
        torch.stack(
            (
                F.log_softmax(original_logits, dim=1),
                F.log_softmax(mirrored_logits, dim=1),
            ),
            dim=0,
        ),
        dim=0,
    ) - math.log(2.0)
    ensemble_nll = F.nll_loss(ensemble_log_probs, labels)
    ensemble_smooth = -ensemble_log_probs.mean(dim=1).mean()
    ensemble_loss = 0.98 * ensemble_nll + 0.02 * ensemble_smooth
    return 0.75 * per_view_loss + 0.25 * ensemble_loss
>>>>>>> REPLACE
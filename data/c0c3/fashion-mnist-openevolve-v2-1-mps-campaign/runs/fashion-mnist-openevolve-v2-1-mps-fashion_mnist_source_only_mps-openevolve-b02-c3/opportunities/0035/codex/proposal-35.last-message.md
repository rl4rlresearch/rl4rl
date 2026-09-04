MECHANISM: Qualified paired-view supervision with calibrated posterior ensembling

HYPOTHESIS: Removing explicit flip-consistency regularization and restoring temperature 0.75 will recover 9,315 correct predictions and approximately 0.19136 cross-entropy.

INTENDED_EDIT: Revert the harmful KL consistency term to paired-view cross-entropy and sharpen the inference ensemble from 0.78 to 0.75.

EVIDENCE: The consistency-regularized current design fell to 9,311 correct, while the otherwise matching Reference Design 1 achieved the best verified score with 9,315 correct and 0.1913601 cross-entropy.

<<<<<<< SEARCH
        ).log() / 0.78
=======
        ).log() / 0.75
>>>>>>> REPLACE

<<<<<<< SEARCH
    paired_logits = model(paired_images)
    first_logits, second_logits = paired_logits.chunk(2, dim=0)

    supervised_loss = F.cross_entropy(
        paired_logits,
        paired_labels,
        label_smoothing=0.02,
    )
    first_log_probabilities = F.log_softmax(first_logits, dim=1)
    second_log_probabilities = F.log_softmax(second_logits, dim=1)
    consistency_loss = 0.5 * (
        F.kl_div(
            first_log_probabilities,
            second_log_probabilities.exp().detach(),
            reduction="batchmean",
        )
        + F.kl_div(
            second_log_probabilities,
            first_log_probabilities.exp().detach(),
            reduction="batchmean",
        )
    )
    return supervised_loss + 0.05 * consistency_loss
=======
    return F.cross_entropy(
        model(paired_images),
        paired_labels,
        label_smoothing=0.02,
    )
>>>>>>> REPLACE